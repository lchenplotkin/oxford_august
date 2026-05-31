import os.path
from steamroller import Environment
from SCons.Environment import OverrideEnvironment

vars = Variables("custom.py")
vars.AddVariables(
    ("EXPERIMENTS", "Definitions of the experiments to run", []),
    ("DATA_PATH", "Root directory containing the Chaucer corpus", "scansionRNN/full_dataset"),
    ("MAX_EPOCHS", "", 5),
    ("EMBEDDING_SIZE", "", 16),
    ("HIDDEN_SIZE", "", 16),
    ("BATCH_SIZE", "", 8),
    ("TRAIN_PROPORTION", "", 0.8),
    ("DEV_PROPORTION", "", 0.1),
    ("TEST_PROPORTION", "", 0.1),
    ("DEVICE", "", "cpu"),
    ("LIMIT", "Limit train/dev/test to this number of lines during training", None),
    ("FOLD_COUNT", "How many randomized experiments to run per experimental configuration", 1),
    ("FOLD", "For tracking the current fold", 0),
    EnumVariable("GRANULARITY", "", "word_scan", ["word_scan", "word_stress", "char_scan", "char_stress"]),
)

env = Environment(
    variables=vars,
    BUILDERS={
        "PreprocessData" : Builder(
            action="python scripts/preprocess_data.py --data_path ${DATA_PATH} --output ${TARGETS[0]}"
        ),
        "SplitData" : Builder(
            action="python scripts/split_data.py --input ${SOURCES[0]} --outputs ${TARGETS} --proportions ${TRAIN_PROPORTION} ${DEV_PROPORTION} ${TEST_PROPORTION} --seed ${FOLD}"
        ),
        "TrainModel" : Builder(
            action="python scripts/train_model.py --train ${SOURCES[0]} --dev ${SOURCES[1]} --test ${SOURCES[2]} --output ${TARGETS[0]} --seed ${FOLD} --max_epochs ${MAX_EPOCHS} --embedding_size ${EMBEDDING_SIZE} --hidden_size ${HIDDEN_SIZE} --batch_size ${BATCH_SIZE} ${'--device ' + DEVICE if DEVICE else ''} --granularity ${GRANULARITY} ${'--limit ' + str(LIMIT) if LIMIT else ''}"
        ),
        "ApplyModel" : Builder(
            action="python scripts/apply_model.py --model ${SOURCES[0]} --inputs ${SOURCES[1:]} --output ${TARGETS[0]} --device ${DEVICE}"
        ),
        "AnalyzeResults" : Builder(
            action="python scripts/analyze_results.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        )        
    }
)


def expand_spec(spec, cur=[{}]):
    k, v = spec[0]
    if isinstance(v, (int, str, float, tuple)):
        next_cur = [c | {k : v} for c in cur]
    elif isinstance(v, list):
        next_cur = sum([[c | {k : val} for c in cur] for val in v], [])
    else:
        raise Exception()
    if len(spec) <= 1:
        return next_cur
    else:
        return expand_spec(spec[1:], next_cur)


data = env.PreprocessData(
    "work/data.jsonl.gz",
    []
)
env.Alias("preprocess", data)

splits = {}

for spec in env["EXPERIMENTS"]:
    for exp in list(expand_spec(sorted(spec.items()))):
        exp_env = env.Override(exp)

        for fold in range(exp_env["FOLD_COUNT"]):
            exp_env = exp_env.Override({"FOLD" : fold})

            if fold not in splits:
                splits[fold] = exp_env.SplitData(
                    ["work/${FOLD}/train.jsonl.gz", "work/${FOLD}/dev.jsonl.gz", "work/${FOLD}/test.jsonl.gz"],
                    data
                )
                exp_env.Alias("split", splits[fold])
                
            train, dev, test = splits[fold]                

            model = exp_env.TrainModel(
                "work/${FOLD}/${NAME}/model.bin",
                [train, dev, test]
            )
            exp_env.Depends(model, "scripts/${GRANULARITY}.py")
            exp_env.Alias("train", model)
            
            output = exp_env.ApplyModel(
                "work/${FOLD}/${NAME}/output.jsonl.gz",
                [model, train, dev, test],
            )
            exp_env.Alias("apply", output)
            
            analysis = exp_env.AnalyzeResults(
              "work/${FOLD}/${NAME}/analysis.txt",
              output
            )
            exp_env.Alias("analyze", analysis)
