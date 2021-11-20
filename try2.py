# %%
import pickle
from text_classifier_len.model_evaluation import ModelEvaluator
from text_classifier_len.model_evaluation import create_and_train_len

# %%


# %%
evaluator = None
with open("evaluator_data", "rb") as f:
    evaluator = pickle.load(f)

# %%
x_train = evaluator.x_train
y_train = evaluator.y_train

# %%
model = create_and_train_len(
    x_train=x_train,
    y_train=y_train,
    batch_size=512,
    learning_rate=0.005,
    num_epochs=50,
    save_the_model=True,
    model_path="model",
    use_cuda=True,
    load_model=True,
)

# %%

