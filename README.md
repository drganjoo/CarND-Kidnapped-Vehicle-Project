# Overview

## Project Introduction
Your robot has been kidnapped and transported to a new location! Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, and lots of (noisy) sensor and control data.

In this project a 2 dimensional particle filter has been implemented in C++. The particle filter has been given a map and some initial localization information (analogous to what a GPS would provide). At each time step the filter gets an observation and control data. 

## Running the Code
This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

## Code Explanation

Number of particles: 5

### Initialization

```
    normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (unsigned int i = 0; i < NUM_PARTICLES; i++) {
		Particle p(i, dist_x(gen), dist_y(gen), dist_theta(gen));
		particles_.push_back(p);
	}

```

### Prediction

When yaw_rate is 0 (predictWithoutYaw)

```
  for (auto &p : particles_) {
    const double v_dt = velocity * delta_t;

		p.x += v_dt * cos(p.theta);
		p.y += v_dt * sin(p.theta);
  }
```

When yaw_rate is > 0 (predictWithYawRate):

```
  for (auto &p : particles_) {
    const double v_dt = velocity * delta_t;

		p.x += v_dt * cos(p.theta);
		p.y += v_dt * sin(p.theta);
  }
```

Random gaussian noise is added in both cases:

```
  // add random gaussian noise to each particle's X & Y position
  random_device rd;
  mt19937 gen(rd());

  for (auto &p : particles_) {
    normal_distribution<double> dist_x(p.x, std_pos[0]);
    normal_distribution<double> dist_y(p.y, std_pos[1]);
    normal_distribution<double> dist_theta(p.theta, std_pos[2]);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
```

### Weight Update

```
for (auto &p : particles_) {
    p.associations.clear();

    for (LandmarkObs &obs : observations) {
      Association nearest_landmark = findNearestLandmark(p, obs, sensor_range, map);
      p.associations.push_back(nearest_landmark);

      const double obs_weight = multivariateGaussian(nearest_landmark.obs_in_ws,
                                                     nearest_landmark.landmark_location,
                                                     std_landmark);
      p.weight *=  obs_weight;
    }
}
```