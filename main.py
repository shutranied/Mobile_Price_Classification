from config import TRAIN_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, MODEL_PATH
from data.data_loader import load_data
from utils.data_exploration import display_basic_info
from utils.visualization import (
    plot_class_balance,
    plot_ram_distribution,
    plot_battery_power_distribution,
    plot_three_g_support
)
from preprocessing.preprocessing import (
    split_features_target,
    scale_features,
    split_train_test
)
from models.ml_models import train_logistic_regression, train_svc_model, save_model
from evaluation.evaluation import evaluate_model, plot_confusion_matrix


def main():
    # Load dataset
    df_train = load_data(TRAIN_DATA_PATH)

    # Display basic dataset information
    display_basic_info(df_train)

    # Visualize selected features
    plot_class_balance(df_train, TARGET_COLUMN)
    plot_ram_distribution(df_train, TARGET_COLUMN)
    plot_battery_power_distribution(df_train, TARGET_COLUMN)
    plot_three_g_support(df_train)

    # Split features and target
    X, y = split_features_target(df_train, TARGET_COLUMN)

    # Scale features
    X_scaled, scaler = scale_features(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = split_train_test(
        X_scaled,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Train Logistic Regression model
    logistic_model = train_logistic_regression(X_train, y_train)
    print("\nLogistic Regression Evaluation:")
    evaluate_model(logistic_model, X_test, y_test)
    plot_confusion_matrix(logistic_model, X_test, y_test)

    # Train SVC model
    svc_model = train_svc_model(X_train, y_train)
    print("\nSupport Vector Classifier Evaluation:")
    evaluate_model(svc_model, X_test, y_test)
    plot_confusion_matrix(svc_model, X_test, y_test)

    # Save final model
    save_model(svc_model, MODEL_PATH)
    print(f"\nModel saved successfully at {MODEL_PATH}")


if __name__ == "__main__":
    main()
