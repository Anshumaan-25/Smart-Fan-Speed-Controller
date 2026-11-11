"""Controller algorithms and reproducible simulator data."""

from dataclasses import dataclass
from math import isfinite, tanh
from random import Random

TEMP_MIN, TEMP_MAX = 18, 42
TEMP_THRESHOLDS = (24, 30)
RATES = (0.01, 0.08)
EPOCHS = 100
HIDDEN_UNITS = 6
TEMP_LABELS = ("Cool", "Warm", "Hot")
HUMIDITY_LABELS = ("Dry", "Comfortable", "Humid")
SPEED_LABELS = ("Very low", "Low", "Medium", "High", "Very high")
RULES = (
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
)
OUTPUT_SETS = tuple(
    tuple(max(0, 1 - abs(speed - 25 * index) / 25) for index in range(len(SPEED_LABELS)))
    for speed in range(101)
)
TRAINING_CASES = tuple((t, h) for t in range(TEMP_MIN, TEMP_MAX + 1, 2) for h in range(0, 101, 10))
HELD_OUT_CASES = tuple((t, h) for t in range(TEMP_MIN + 1, TEMP_MAX, 2) for h in range(5, 100, 10))


def normalize(temperature, humidity):
    if not TEMP_MIN <= temperature <= TEMP_MAX or not 0 <= humidity <= 100:
        raise ValueError(f"Temperature must be {TEMP_MIN}–{TEMP_MAX} °C and humidity must be 0–100%.")
    return (temperature - TEMP_MIN) / (TEMP_MAX - TEMP_MIN), humidity / 100


def reference_speed(temperature, humidity):
    """Simulated target. Humidity matters more as the room gets hotter."""
    heat, moisture = normalize(temperature, humidity)
    return 100 * (0.05 + 0.63 * heat + 0.12 * moisture + 0.16 * heat * moisture)


def crisp_speed(temperature, humidity):
    normalize(temperature, humidity)
    warm, hot = TEMP_THRESHOLDS
    base = 20 if temperature < warm else 50 if temperature < hot else 75
    return min(100, base + (15 if humidity >= 70 else 0))


def memberships(value):
    """Left shoulder, triangle, and right shoulder over the unit interval."""
    return max(0, 1 - 2 * value), 1 - abs(2 * value - 1), max(0, 2 * value - 1)


@dataclass(frozen=True)
class FuzzyResult:
    speed: float
    firing: tuple
    aggregate: tuple
    strongest: tuple


def fuzzy_inference(temperature, humidity):
    """Mamdani inference with min clipping, max aggregation, and centroid."""
    heat, moisture = normalize(temperature, humidity)
    firing = tuple(
        tuple(min(heat_degree, moisture_degree) for moisture_degree in memberships(moisture))
        for heat_degree in memberships(heat)
    )
    activations = [0.0] * len(SPEED_LABELS)
    strongest = (0.0, 0, 0, 0)
    for row, strengths in enumerate(firing):
        for column, strength in enumerate(strengths):
            speed = RULES[row][column]
            activations[speed] = max(activations[speed], strength)
            if strength > strongest[0]:
                strongest = strength, row, column, speed

    aggregate = tuple(
        max(min(strength, degree) for strength, degree in zip(activations, output_set))
        for output_set in OUTPUT_SETS
    )
    centroid = sum(speed * degree for speed, degree in enumerate(aggregate)) / sum(aggregate)
    return FuzzyResult(centroid, firing, aggregate, strongest)


def fuzzy_speed(temperature, humidity):
    result = fuzzy_inference(temperature, humidity)
    return result.speed, result.strongest


class NeuralController:
    """A 2–6–1 MLP with tanh hidden units and a linear output."""

    def __init__(self):
        random = Random(21)
        self.hidden = [
            [random.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(HIDDEN_UNITS)
        ]
        self.output = [random.uniform(-0.5, 0.5) for _ in range(HIDDEN_UNITS)]
        self.bias = 0.5

    def forward(self, heat, moisture):
        hidden = [tanh(w1 * heat + w2 * moisture + bias) for w1, w2, bias in self.hidden]
        prediction = self.bias + sum(weight * value for weight, value in zip(self.output, hidden))
        return hidden, prediction

    def speed(self, temperature, humidity):
        _, prediction = self.forward(*normalize(temperature, humidity))
        return 100 * min(1, max(0, prediction))

    def mean_squared_error(self, samples):
        return sum(
            (self.forward(heat, moisture)[1] - target) ** 2
            for heat, moisture, target in samples
        ) / len(samples)

    def train(self, samples, learning_rate):
        samples = tuple(samples)
        if not samples:
            raise ValueError("Training requires at least one sample.")
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive.")
        random = Random(42)
        order = list(samples)
        history = [self.mean_squared_error(samples)]
        for _ in range(EPOCHS):
            random.shuffle(order)
            for heat, moisture, target in order:
                hidden, prediction = self.forward(heat, moisture)
                output_delta = prediction - target  # Gradient of half squared error.
                for index, value in enumerate(hidden):
                    hidden_delta = output_delta * self.output[index] * (1 - value**2)
                    weights = self.hidden[index]
                    weights[0] -= learning_rate * hidden_delta * heat
                    weights[1] -= learning_rate * hidden_delta * moisture
                    weights[2] -= learning_rate * hidden_delta
                    self.output[index] -= learning_rate * output_delta * value
                self.bias -= learning_rate * output_delta
            history.append(self.mean_squared_error(samples))
        return history


def run_experiment():
    training = [
        (*normalize(temperature, humidity), reference_speed(temperature, humidity) / 100)
        for temperature, humidity in TRAINING_CASES
    ]
    models = [NeuralController() for _ in RATES]
    losses = [model.train(training, rate) for model, rate in zip(models, RATES)]
    # Midpoints between training grid points. This grid is used only for evaluation.
    predictors = [crisp_speed, lambda t, h: fuzzy_speed(t, h)[0], *(m.speed for m in models)]
    errors = [
        sum(abs(predict(t, h) - reference_speed(t, h)) for t, h in HELD_OUT_CASES)
        / len(HELD_OUT_CASES)
        for predict in predictors
    ]
    return models, losses, errors
