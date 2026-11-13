# Smart Fan Speed Controller

A Soft Computing mini-project for Modules 1–3. The desktop dashboard compares a crisp threshold controller, a Mamdani fuzzy controller, and a backpropagation neural network for the same room conditions. The sliders cover 18–42 °C and 0–100% relative humidity.

## Run the simulator

On this Mac, double-click **Launch-Fan-Lab.command** in Finder, or run:

```sh
cd "/Users/anshu/Documents/PROJECTS/SC project "
./Launch-Fan-Lab.command
```

The launcher uses a Python installation with Tk 8.6 or newer. On this Mac, the older system Tk can open a blank window, so the launcher can use the installed Miniconda Python. To run directly on another computer, open the project folder and run `python3 fan_lab.py` with Python 3.9 or newer and Tk 8.6 or newer. The project needs no third-party Python packages, hardware, or external dataset.

## Demonstrate it

1. Open **Temperature sweep** and click **29.9 °C**, then **30.1 °C**. Both presets use 60% relative humidity. The crisp controller jumps from 50% to 75%, while the fuzzy and neural predictions each move by less than one percentage point. The vertical marker follows the current temperature. The dashed line is the simulated target at the current humidity.
2. Open **Fuzzy rules** and choose **Hot / humid**. The rule grid shows all nine firing strengths and highlights the strongest rule or tied rules. The plot below shows the aggregated membership function and the centroid used for the final speed.
3. Open **Neural training** and switch between learning rates **0.01** and **0.08**. Both curves start at the same epoch-zero loss. The selector changes which trained network supplies the live speed and held-out error.
4. Choose any controller in **Fan preview** to see its speed drive the animation. Use **Pause** to freeze the blades while explaining the results.

## Module 1 · Crisp thresholds

The base speed is 20% below 24°C, 50% from 24°C up to 30°C, and 75% at 30°C or above. At 70% relative humidity or above, the controller adds 15 percentage points. Small input changes across a boundary can cause an abrupt speed change.

## Module 2 · MLP and backpropagation

The network takes two normalized inputs, uses six tanh hidden units, and predicts one speed. It learns with stochastic gradient descent. Backpropagation computes the output error, propagates it through the hidden weights and tanh derivative, and updates the weights. The two runs share their initial weights, training sample order, and 100-epoch budget. The chart plots training mean squared error (MSE) on normalized speed on a log scale. The held-out MAE on the controller card uses percentage points.

## Module 3 · Fuzzy controller

Both inputs have three overlapping membership functions. The outer sets use shoulders, and the middle set uses a triangle. Each rule uses the minimum of the two input memberships as its firing strength. The controller clips the output sets, aggregates them with maximum, and takes the centroid over the integer speeds from 0% through 100%.

| Temperature | Dry | Comfortable | Humid |
| --- | --- | --- | --- |
| Cool | Very low | Low | Medium |
| Warm | Low | Medium | High |
| Hot | Medium | High | Very high |

The five output sets peak at 0%, 25%, 50%, 75%, and 100%, respectively.

## Evaluation and scope

The reference is a synthetic comfort policy. For normalized temperature `t = (temperature − 18) / 24` and normalized humidity `h = relative humidity / 100`, it sets the target fan speed to `100 × (0.05 + 0.63t + 0.12h + 0.16th)`. The interaction term makes humidity count more at higher temperatures.

Training uses 143 points on a regular grid. All three methods are evaluated on 120 held-out grid midpoints using mean absolute error (MAE) in percentage points. The neural network is trained to reproduce the same reference policy used for evaluation, so these errors measure interpolation of a simulated policy. Physical fan performance and human comfort would require measured data.

| Controller | Held-out MAE |
| --- | ---: |
| Crisp thresholds | 13.79 pp |
| Fuzzy inference | 9.96 pp |
| MLP, learning rate 0.01 | 0.98 pp |
| MLP, learning rate 0.08 | 0.94 pp |

`controllers.py` contains the algorithms and experiment. `fan_lab.py` contains the desktop interface.

## Run the controller checks

```sh
python3 -m unittest -v
```


:)

