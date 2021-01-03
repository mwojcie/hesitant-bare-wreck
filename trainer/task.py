from trainer.model import get_classifier
from trainer.utils import prepare_training_data
from sklearn.metrics import classification_report

DATA_DIRECTORY = "../data/"

x_train, x_test, y_train, y_test = prepare_training_data(DATA_DIRECTORY)
classifier = get_classifier()
classifier.fit(x_train, y_train)
y_hat = classifier.predict(x_test)
report = classification_report(y_test, y_hat, output_dict=True)
print(report)
