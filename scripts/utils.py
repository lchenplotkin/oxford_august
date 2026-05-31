import json
from itertools import takewhile
import torch
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torch.nn.utils import clip_grad_norm_

def run_epoch(model, dataloader, optimizer=None, out=None, compute_loss=True):
    if optimizer:
        model.train()
    else:
        model.eval()
    losses = []
    count = 0
    total_stress_chances = 0
    correct_stresses = 0
    total_scan_chances = 0
    correct_scans = 0
    for batch in dataloader:
        probs = model(batch)
        scan_level = len(probs.shape) == 3
        count += batch["length_in_words"].sum()

        mask = batch["scan_mask" if scan_level else "stress_mask"]
        target = batch["scans" if scan_level else "stresses"]

        ce_input = torch.transpose(probs, 1, -1)
        ce_target = torch.transpose(target, 1, -1)
        ces = torch.transpose(cross_entropy(ce_input, ce_target, reduction="none"), 1, -1)
        loss = ces[mask].sum()

        if optimizer:
            loss.backward()
            clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()
        losses.append(loss.detach().item())

        predictions = torch.argmax(probs, dim=-1).detach()

        for i, prediction in enumerate(predictions):
            item = dataloader._dataset._items[batch["indices"][i]]
            item["predictions"] = []
            annotated = len(item["scans"]) == len(item["words"])
            for w, word in enumerate(prediction):
                if w >= len(item["words"]):
                    break
                if scan_level:
                    scan_id = word.item()
                    scan = dataloader._dataset._id2scan[scan_id]
                    preds = [] if scan_id == 0 else list(scan)
                else:
                    preds = [dataloader._dataset._id2stress[j] for j in takewhile(lambda x : x != 1, word.tolist())]
                item["predictions"].append(preds)
                
                if not annotated:
                    continue
                golds = list(item["scans"][w])                
                total_scan_chances += 1
                if preds == golds:
                    correct_scans += 1
                total_stress_chances += max([len(preds), len(golds)])
                for a, b in zip(preds, golds):
                    if a == b:
                        correct_stresses += 1
            
            if out:
                out.write(json.dumps(item) + "\n")
                        

    return {
        "loss" : sum(losses) / count,
        "stress_accuracy" : correct_stresses / total_stress_chances,
        "scan_accuracy" : correct_scans / total_scan_chances
    }
