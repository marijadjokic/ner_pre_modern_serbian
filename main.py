# #!/usr/bin/env python

# ── 1.  EDIT ONLY THESE THREE PATHS ─────────────────────────────
MODEL_DIR = "trained_models/output/model-best"              # spaCy model
DOCX_PATH = "docs/testing/church_document_annotated.docx"   # Word file
JSON_PATH = "docs/testing/church_document_annotated.json"   # gold spans

# ───────────────────────────────────────────────────────────────

import json
from pathlib import Path

import docx
import spacy
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text_from_docx(path: Path) -> str:
    """
    Extract the full text of a DOCX file, keeping a single '\n' between
    paragraphs so that character offsets remain stable.
    """
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_spans(path: Path):
    """
    Return a list of (start, end, label) tuples from a JSON file with
    structure

        {
          "entities": [
              [42,  48, "PERSON"],
              [49,  62, "PERSON"],
              ...
          ]
        }
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [tuple(triple) for triple in data["entities"]]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(nlp, text: str, gold_spans):
    """
    Print every entity the model finds, tagged with its outcome
    (TP / FP).  Also list the gold entities the model missed (FN).
    Finally print standard classification metrics.
    """
    # 1) run the model
    doc  = nlp(text)
    pred = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

    # 2) match predictions ↔ gold
    gold_set      = {(s, e, lbl) for (s, e, lbl) in gold_spans}
    matched_preds = set()                    # indices of predictions → TP
    pred_status   = ["FP"] * len(pred)       # default every prediction to FP

    for i, (p_s, p_e, p_lbl) in enumerate(pred):
        if (p_s, p_e, p_lbl) in gold_set:
            pred_status[i] = "TP"
            matched_preds.add(i)

    fn_entities = [
        (g_s, g_e, g_lbl)
        for (g_s, g_e, g_lbl) in gold_spans
        if (g_s, g_e, g_lbl) not in gold_set.intersection(gold_set)  # same form
           and (g_s, g_e, g_lbl) not in [(p_s, p_e, p_lbl) for (p_s, p_e, p_lbl) in pred]
    ]

    # 3) pretty-print outcomes in reading order
    print("\n=== Entity-level outcomes (reading order) ===")
    for idx, (s, e, lbl) in enumerate(pred):
        print(f"{pred_status[idx]:2}  [{s:>6}, {e:<6}]  {text[s:e]!r}  ({lbl})")

    for s, e, lbl in fn_entities:
        print(f"FN  [{s:>6}, {e:<6}]  {text[s:e]!r}  ({lbl})")

    # 4) build lists for sklearn
    y_true, y_pred = [], []

    #   gold → TP / FN
    for (g_s, g_e, g_lbl) in gold_spans:
        y_true.append(g_lbl)
        if (g_s, g_e, g_lbl) in gold_set.intersection({(p_s, p_e, p_lbl) for (p_s, p_e, p_lbl) in pred}):
            y_pred.append(g_lbl)                   # TP
        else:
            y_pred.append("O")                     # FN

    #   remaining predictions → FP
    for i, (p_s, p_e, p_lbl) in enumerate(pred):
        if i not in matched_preds:
            y_true.append("O")
            y_pred.append(p_lbl)

    # 5) metrics
    print("\n=== classification report ===")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    labels = sorted({l for l in y_true + y_pred if l != "O"}) + ["O"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"T_{l}" for l in labels],
        columns=[f"P_{l}" for l in labels]
    )
    print("\n=== confusion matrix ===")
    print(cm_df)

    print("\n=== TP / FP / FN per label ===")
    summary = []
    for i, lbl in enumerate(labels):
        if lbl == "O":
            continue
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        summary.append([lbl, TP, FP, FN])
    print(
        pd.DataFrame(summary, columns=["label", "TP", "FP", "FN"])
        .set_index("label")
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading spaCy model …")
    nlp = spacy.load(MODEL_DIR)

    print("Reading DOCX …")
    text = text_from_docx(Path(DOCX_PATH))
    print(f"  {len(text):,d} characters extracted")

    print("Loading gold spans …")
    gold_spans = load_spans(Path(JSON_PATH))
    print(f"  {len(gold_spans)} gold entities")

    print("Running evaluation …")
    evaluate(nlp, text, gold_spans)
