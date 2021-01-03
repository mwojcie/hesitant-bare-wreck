import logging

from sklearn.metrics import classification_report

from trainer.model import get_classifier
from trainer.utils import DATA_DIRECTORY
from trainer.utils import prepare_training_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

x_train, x_test, y_train, y_test = prepare_training_data(DATA_DIRECTORY)
classifier = get_classifier()
logger.info("Training the model")
classifier.fit(x_train, y_train)
y_hat = classifier.predict(x_test)
logger.info(classification_report(y_test, y_hat, output_dict=True))
