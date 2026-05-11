from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using classification report.
    """
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred


def plot_confusion_matrix(model, X_test, y_test):
    """
    Display the confusion matrix for the trained model.
    """
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.show()
