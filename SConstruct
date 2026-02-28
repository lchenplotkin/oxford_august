import os.path
from steamroller import Environment
from SCons.Environment import OverrideEnvironment

vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("RANDOM_SEED", "", 0),
    ("DATA_PATH", "", "scansionRNN/full_dataset"),
    ("RANDOM_SEED", "", 0),
    ("MAX_EPOCHS", "", 5),
    ("EMBEDDING_SIZE", "", 16),
    ("HIDDEN_SIZE", "", 16),
    ("BATCH_SIZE", "", 8),
    ("CHARACTER_LEVEL", "", False),
    ("TRAIN_PROPORTION", "", 0.8),
    ("DEV_PROPORTION", "", 0.1),
    ("TEST_PROPORTION", "", 0.1),
    ("DEVICE", "", "cpu"),
    ("MODEL_TYPE", "", "lstm"),
    ("EXPERIMENTS", "", [{"NAME" : "example"}])
)

env = Environment(
    variables=vars,
    BUILDERS={
        "PreprocessData" : Builder(
            action="python scripts/preprocess_data.py --data_path ${DATA_PATH} --outputs ${TARGETS[0]} ${TARGETS[1]} ${TARGETS[2]} --proportions ${TRAIN_PROPORTION} ${DEV_PROPORTION} ${TEST_PROPORTION} --seed ${RANDOM_SEED}"
        ),
        "TrainModel" : Builder(
            action="python scripts/train_model.py --train ${SOURCES[0]} --dev ${SOURCES[1]} --test ${SOURCES[2]} --output ${TARGETS[0]} --seed ${RANDOM_SEED} --max_epochs ${MAX_EPOCHS} --embedding_size ${EMBEDDING_SIZE} --hidden_size ${HIDDEN_SIZE} --batch_size ${BATCH_SIZE} ${'--device ' + DEVICE if DEVICE else ''}"
        ),
        "ApplyModel" : Builder(
            action="python scripts/apply_model.py --model ${SOURCES[0]} --inputs ${SOURCES[1:]} --output ${TARGETS[0]}"
        ),
        "AnalyzeResults" : Builder(
            action="python scripts/analyze_results.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        )        
    }
)


for exp in env["EXPERIMENTS"]:
    exp_env = env.Override(exp)
    train, dev, test = exp_env.PreprocessData(
        ["work/${NAME}/train.jsonl.gz", "work/${NAME}/dev.jsonl.gz", "work/${NAME}/test.jsonl.gz"],
        []
    )
    model = exp_env.TrainModel(
        "work/${NAME}/model.bin",
        [train, dev, test]
    )
    output = exp_env.ApplyModel(
        "work/${NAME}/output.jsonl.gz",
        [model, train, dev, test]
    )
    analysis = exp_env.AnalyzeResults(
        "work/${NAME}/analysis.txt",
        output
    )
