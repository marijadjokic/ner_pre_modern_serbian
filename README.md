# Serbian Diplomatic Minuscule Cyrillic NER Evaluation

This project provides an evaluation script for Named Entity Recognition (NER) models trained on Serbian diplomatic minuscule Cyrillic documents. It is designed to work with spaCy models and annotated DOCX/JSON files.

## Project Structure

- `main.py` — Main evaluation script
- `trained_models/output/model-best/` — Directory containing the trained spaCy model
- `docs/testing/` — Contains annotated DOCX and JSON files for evaluation

## Requirements

- Python 3.7+
- [spaCy](https://spacy.io/)
- [python-docx](https://python-docx.readthedocs.io/)
- pandas
- scikit-learn

Install dependencies with:

```bash
pip install spacy python-docx pandas scikit-learn
```

## Usage

1. Edit the paths at the top of `main.py` to point to your model, DOCX, and JSON files:
   - `MODEL_DIR` — Path to the trained spaCy model
   - `DOCX_PATH` — Path to the annotated DOCX file
   - `JSON_PATH` — Path to the gold standard JSON file

2. Run the evaluation script:

```bash
python main.py
```

The script will print out entity-level outcomes, a classification report, and a confusion matrix.

## Data Format

- **DOCX**: The document to be evaluated, with text in Serbian diplomatic minuscule Cyrillic.
- **JSON**: Gold standard entity spans, with structure:

```json
{
  "entities": [
    [42, 48, "PERSON"],
    [49, 62, "PERSON"],
    ...
  ]
}
```

## Training Code Access

If you wish to access the full codebase for training your own NER model on Serbian diplomatic minuscule Cyrillic, please contact the authors directly. The training scripts and data preparation tools are not included in this repository.

## Authors

- [Marija Đokić Petrović](https://marijadjokicpetrovic.com/)
- [Vladimir Polomac](https://en.kg.ac.rs/teachers_teacher.php?fakultet_je=11&nast_je=82)
- [Mihailo St. Popović](http://www.oeaw.ac.at/oeaw/staff/popovic-mihailo)

## License

See `LICENSE` for details.
