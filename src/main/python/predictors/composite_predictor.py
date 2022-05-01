from python.predictor import Predictor


def sign(number):
    if number > 0:
        return 1
    elif number < 0:
        return -1
    else:
        return 0


def da(actual, predicted):
    n = len(actual)
    result = 0

    for i in range(1, n):
        actual_sign = sign(actual[i] - actual[i - 1])
        predicted_sign = sign(predicted[i] - predicted[i - 1])
        if actual_sign == predicted_sign:
            result += 1

    return result


def rae(actual, predicted):
    n = len(actual)
    result = 0

    for i in range(n):
        result += abs(actual[i] - predicted[i])

    return result


class CompositePredictor(Predictor):
    def __init__(self, error_window, predictors):
        self.error_window = error_window
        self.predictors = predictors
        self.prediction_history = None
        self.errors = {p.short_name(): [] for p in predictors}

    def name(self):
        return "Composite"

    def short_name(self):
        return "C"

    def __update_errors__(self, history, t):
        actual = history[-t-1:-1]

        if self.prediction_history is None:
            self.prediction_history = {p.short_name(): [] for p in self.predictors}
        else:
            for name, predictions in self.prediction_history.items():
                predicted = predictions[-t:]
                error = rae(actual, predicted)
                errors = self.errors[name]
                errors.append(error)

                if len(errors) - 1 == self.error_window:
                    errors.pop(0)

    def __find_best_predictor__(self):
        min_error = 1e16
        min_name = None

        for name, errors in self.errors.items():
            sum_error = sum(errors)

            if sum_error < min_error:
                min_error = sum_error
                min_name = name

        return min_name

    def get_prediction(self, history, t=1):
        self.__update_errors__(history, t)

        predictions = {p.short_name(): p.get_prediction(history, t) for p in self.predictors}

        for name, prediction in self.prediction_history.items():
            prediction.extend(predictions[name])

        best_predictor = self.__find_best_predictor__()
        print(f"Choice: {best_predictor}")
        return predictions[best_predictor]
