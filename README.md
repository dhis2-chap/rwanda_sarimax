# Rwanda sarimax model integrated with chap 

The repository integrates a locally developed Sarimax model into chap.

This model is under development, but it is possible to already run it through chap.


### How to run this model locally through chap

1. Make sure you have chap installed (see the chap documentation)
2. You will will need to have docker installed
3. Clone this repository
4. Assuming you have this repository somewhere on your computer, you can run evaluations using the `chap evaluate` command.


Example of an evaluation using a chap builtin dataset:

```bash
chap evaluate --model-name ../rwanda_sarimax/ --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=2 --run-directory-type use_existing
```

This will generate a `report.pdf` with results from the evaluation. Note that we specify `--run-directory-type` to be `use_existing`. This means that chap will use the directory for the model as a working directory to put dataset files in. This makes it easier to debug and inspect results. Chap will e.g. put files like training_data.csv, etc in this directory. Skipping this option will mean chap makes a new directory in the run folder for each run.


