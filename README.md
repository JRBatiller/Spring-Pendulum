# Spring-Pendulum

Abstract

A 2D spring pendulum model was created in Python to simulate an octopus arm that acts as a physical reservoir for reservoir computing. Composed of 10 mass nodes and 20 springs, it maps a 1-dimensional input function into a 20-dimensional state space whose weighted sum could be used to emulate the nonlinear auto-regressive moving average (NARMA) system, equations that exhibit nonlinearity as well as time delays. The model’s computational performance as well as its memory capacity were evaluated by examining the mean squared error of the target output and the system output, as different parameters such as spring constants (k) and damping coefﬁcients (d) were changed. It was determined that spring constants in the range of 5-25 N/m paired with damping constants of 1-2 Ns/m minimized the error to around 4.85×10−7. It also appears that this combination of parameters also increases the memory of the system, making it better tolerate delays of up to 1.0 second.

spring_pen_X.py - simulates a spring pendulum with assigned parameters and saves a .mat file containing the spring lengths, spring angles and node coordinates over time t.

Memory_data_gen - this takes the generated pendulum data and attempts to train it to the NARMA series (or volterra) and gives the mean squared error. Memory tests are also done by checking performance after adding an increasing delay to one of the signals.

memory_weights.py, memory_tests.py, heatmap_gen.py - these are for generating plots and figures to make data easier to understand

Not sure if I can post my entire paper here or not...
