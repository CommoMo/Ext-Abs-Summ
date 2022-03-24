import os

from utils import get_logger
from transformers import AdamW, get_linear_schedule_with_warmup
from fastprogress.fastprogress import master_bar, progress_bar

from data_utils import get_dataloader
import torch
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, Softmax
from sklearn.metrics import accuracy_score

from rouge import Rouge

def train(args, model, criterion):
    ### Set Loggers ###
    logger = get_logger(args)
    device = model.model.device.type
    tokenizer = model.tokenizer
    softmax = Softmax(dim=-1)
    
    train_dataloader = get_dataloader(args, tokenizer, data_type='train')

    best_rougel = 0
    global_step = 1
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * args.warmup_proportion), num_training_steps=t_total
    )
    # pos_weight = torch.tensor[3].repeat(args.max_seq_length)
    # criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = BCEWithLogitsLoss()

    steps_trained_in_current_epoch=0
    mb = master_bar(range(int(args.num_train_epochs)))

    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        train_loss = 0
        model.train()
        for step, batch in enumerate(epoch_iterator, 1):
            # Skip past any already trained steps if resuming training
            global_step += 1
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            input_ids, attention_mask, decoder_input_ids, ext_labels, abs_labels, content, target_texts = batch
            input_ids, attention_mask, decoder_input_ids, ext_labels, abs_labels = \
                input_ids.to(device), attention_mask.to(device), decoder_input_ids.to(device), ext_labels.to(device), abs_labels.to(device)
            decoder_attention_mask = decoder_input_ids.ne(3).float().to(device)

            outputs, encoder_classified_token = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, abs_labels)
            
            try:
                decode_loss = outputs.loss
                encode_loss = criterion(encoder_classified_token.squeeze(-1), ext_labels)
            except:
                continue
            loss = encode_loss + decode_loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_loss += loss.item()
            
            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step() # Update learning rate schedule
                model.zero_grad()

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.evaluate_during_training:
                        eval_loss, eval_acc, rouge1, rouge2, rougel = evaluate(args, model, criterion)
                        progress = global_step/t_total

                        logger.info(f"epoch: {epoch}, step: {global_step}, eval_loss: {eval_loss:.4f}, acc: {eval_acc:.4f}, rouge-1: {rouge1:.4f}, rouge-2: {rouge2:.4f}, rouge-l: {rougel:.4f}")

                        decoded_output = tokenizer.decode(torch.topk(softmax(outputs.logits[0]), 1).indices.squeeze(-1), skip_special_tokens=True)
                        logger.info(f"INFER: {decoded_output}")
                        model.train()

             # Save model checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if rougel > best_rougel:
                        best_rougel = rougel
                        output_dir = os.path.join(args.ckpt_dir, args.output_dir, f"checkpoint-{global_step}")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        torch.save(model, os.path.join(output_dir, 'multitask_ext_abs_summary_model.pt'))
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

        mb.write("Epoch {} done".format(epoch+1))

    return global_step, train_loss/global_step

def evaluate(args, model, criterion):
    device = model.model.device.type
    tokenizer = model.tokenizer
    valid_dataloader = get_dataloader(args, tokenizer, data_type='valid')
    
    rouge_scorer = Rouge()
    softmax = Softmax(dim=-1)

    epoch_iterator = progress_bar(valid_dataloader)
    eval_loss = 0
    total_acc = 0
    total_rouge1, total_rouge2, total_rougel = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator, 1):
            input_ids, attention_mask, decoder_input_ids, ext_labels, abs_labels, content, target_texts = batch
            input_ids, attention_mask, decoder_input_ids, ext_labels, abs_labels = \
                input_ids.to(device), attention_mask.to(device), decoder_input_ids.to(device), ext_labels.to(device), abs_labels.to(device)
            decoder_attention_mask = decoder_input_ids.ne(3).float().to(device)

            outputs, encoder_classified_token = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, abs_labels)

            decode_loss = outputs.loss
            try:
                encode_loss = criterion(encoder_classified_token.squeeze(-1), ext_labels)
            except:
                continue

            loss = encode_loss + decode_loss

            eval_loss += loss.item()

            # Encoder scoring
            preds = (encoder_classified_token > 0.5)
            flatten_preds = torch.flatten(preds).tolist()
            flatten_labels = torch.flatten(ext_labels).tolist()
            encoder_acc_score = accuracy_score(flatten_preds, flatten_labels)

            # Decoder scoring
            decoded_label = [' '.join(tokenizer.batch_decode(label, skip_special_tokens=True)) for label in abs_labels]
            tokenized_label = [' '.join(tokenizer.tokenize(label)) for label in decoded_label]
            
            decoded_output = tokenizer.batch_decode(torch.topk(softmax(outputs.logits), 1).indices.squeeze(-1), skip_special_tokens=True)
            tokenized_output = [' '.join(tokenizer.tokenize(output_text)) for output_text in decoded_output]
            
            scores = rouge_scorer.get_scores(tokenized_output, tokenized_label, avg=True)
            rouge1 = scores['rouge-1']['f']
            rouge2 = scores['rouge-2']['f']
            rougel = scores['rouge-l']['f']

            total_acc += encoder_acc_score
            total_rouge1 += rouge1
            total_rouge2 += rouge2
            total_rougel += rougel

    return eval_loss/step, total_acc/step, total_rouge1/step, total_rouge2/step, total_rougel/step