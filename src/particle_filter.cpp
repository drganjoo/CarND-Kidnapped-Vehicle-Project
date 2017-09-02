/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <climits>
#include <cassert>
#include <thread>
#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  random_device rd;
  mt19937 gen(rd());

  normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (unsigned int i = 0; i < NUM_PARTICLES; i++) {
		Particle p(i, dist_x(gen), dist_y(gen), dist_theta(gen));
		particles_.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  assert(particles_.size() == NUM_PARTICLES);

  if (yaw_rate == 0) {
    predictWithoutYaw(delta_t, std_pos, velocity);
  }
  else {
    predictWithYawRate(delta_t, std_pos, velocity, yaw_rate);
  }

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
}

void ParticleFilter::predictWithYawRate(double delta_t, const double *std_pos, double velocity, double yaw_rate) {
  const double vel_per_rate = velocity / yaw_rate;

  for (auto &p : particles_) {
    const double heading_yaw_dt = p.theta + (yaw_rate * delta_t);

    p.x += vel_per_rate * (sin(heading_yaw_dt) - sin(p.theta));
    p.y += vel_per_rate * (cos(p.theta) - cos(heading_yaw_dt));
    p.theta += yaw_rate * delta_t;
  }
}

void ParticleFilter::predictWithoutYaw(double delta_t, const double *std_pos, double velocity) {
  for (auto &p : particles_) {
    const double v_dt = velocity * delta_t;

		p.x += v_dt * cos(p.theta);
		p.y += v_dt * sin(p.theta);
  }
}


Association ParticleFilter::findNearestLandmark(const Particle &p, LandmarkObs &obs, double sensor_range,
                                                const Map &map){
  Association landmark_assoc;
  landmark_assoc.obs_in_ws = p.transformToWorldSpace(obs);
  landmark_assoc.landmark_id = 0;

  // find distances to each land mark
  const auto count = map.landmark_list.size();
  double min_distance = INT_MAX;

  for (const auto &lm : map.landmark_list) {
    const double distance = landmark_assoc.obs_in_ws.getDistance(lm.x_f, lm.y_f);

    if (distance < sensor_range && distance < min_distance) {
      min_distance = distance;

      landmark_assoc.landmark_id = lm.id_i;
      landmark_assoc.landmark_location.x = lm.x_f;
      landmark_assoc.landmark_location.y = lm.y_f;
    }
  }

  assert(landmark_assoc.landmark_id > 0);
  return landmark_assoc;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
																	 std::vector<LandmarkObs> &observations,
																	 const Map &map) {
	for (auto &p : particles_) {
    p.associations.clear();
    p.weight = 1;

    for (LandmarkObs &obs : observations) {
      Association nearest_landmark = findNearestLandmark(p, obs, sensor_range, map);
      p.associations.push_back(nearest_landmark);

      const double obs_weight = multivariateGaussian(nearest_landmark.obs_in_ws,
                                                     nearest_landmark.landmark_location,
                                                     std_landmark);
      p.weight *=  obs_weight;
    }
	}
}

void ParticleFilter::resample() {
	vector<double> weights;
	transform(particles_.begin(), particles_.end(), back_inserter(weights), [](const Particle &p) { return p.weight;  });

  random_device rd;
  mt19937 gen(rd());

	discrete_distribution<> dist(weights.begin(), weights.end());

	vector<Particle> resampled;
	for (unsigned int i = 0; i < NUM_PARTICLES; i++) {
		resampled.push_back(particles_[dist(gen)]);
	}

	// change the particle samples for next time
	particles_ = resampled;
}

string ParticleFilter::getAssociations(const Particle &best) {
  std::stringstream ss;
	std::transform(best.associations.begin(), best.associations.end(), ostream_iterator<double>(ss, " "),
                 [](const Association &a) {
                     return a.landmark_id;
                 });
	std::string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseX(const Particle &best) {
	return getVectorToString(best.associations, [](const Association &a) { return a.obs_in_ws.x; });
}

string ParticleFilter::getSenseY(const Particle &best) {
	return getVectorToString(best.associations, [](const Association &a) { return a.obs_in_ws.y; });
}


double ParticleFilter::multivariateGaussian(const Point &obs_in_ws, const Point &landmark_pt, double *std)
{
	// calculate multivariate Gaussian
	// e = ((x - mu_x) ** 2 / (2.0 * std_x ** 2)) + ((y - mu_y) ** 2 / (2.0 * std_y ** 2))
	// e = ((diff_x ** 2) / (2.0 * std_x ** 2)) + ((diff_y ** 2) / (2.0 * std_y ** 2))

  const double std_x = std[0];
  const double std_y = std[1];

	const double diff_x = obs_in_ws.x - landmark_pt.x;
  const double diff_y = obs_in_ws.y - landmark_pt.y;
  const double diff_x_2 = diff_x * diff_x;
	const double diff_y_2 = diff_y * diff_y;

	const double var_x = std_x * std_x;
	const double var_y = std_y * std_y;

	const double exponent = diff_x_2 / (2.0 * var_x) + diff_y_2 / (2.0 * var_y);
  const double exp_exponent = exp(-exponent);

	// gauss_norm = (1.0 / (2.0 * pi * std_x * std_y))
	const double gauss_norm = 1.0 / (2.0 * M_PI * std_x * std_y);
	return gauss_norm * exp_exponent;
}
