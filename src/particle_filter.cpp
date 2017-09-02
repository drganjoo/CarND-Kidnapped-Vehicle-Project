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

#include "particle_filter.h"

using namespace std;


class ExecTime
{
public:
    ExecTime() {
      struct timespec spec;
      clock_gettime(CLOCK_MONOTONIC, &spec);
      start  = spec.tv_sec;
      start_us = spec.tv_nsec / 1.0e3;    // microseconds
    }

    double End() {
      struct timespec spec;
      clock_gettime(CLOCK_MONOTONIC, &spec);

      end  = spec.tv_sec;
      end_us = spec.tv_nsec / 1.0e3;    // microseconds

      double diff;
      if (end > start) {
        // from the first second choose as many microseconds as they were left in it
        // 4.7 seconds to 5.1 seconds is 400ms so we take 300ms from first

        diff = 1000 * 1000 - start_us;
        diff += end_us + (end - start - 1);
      }
      else
        diff = end_us - start_us;

      return diff;
    }

private:
    time_t start, end;
    double start_us, end_us;
};

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
  ExecTime timing;

  if (yaw_rate == 0) {
    Predict(delta_t, std_pos, velocity);
  }
  else {
    PredictWithYawRate(delta_t, std_pos, velocity, yaw_rate);
  }

  // add rangome gaussian noise to each particle's X & Y position
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

  cout << "Time taken prediction: " << fixed << timing.End() << endl;
}

void ParticleFilter::PredictWithYawRate(double delta_t, const double *std_pos, double velocity, double yaw_rate) {
  const double vel_per_rate = velocity / yaw_rate;

  for (auto &p : particles_) {
    const double heading_yaw_dt = p.theta + (yaw_rate * delta_t);

    p.x += vel_per_rate * (sin(heading_yaw_dt) - sin(p.theta));
    p.y += vel_per_rate * (cos(p.theta) - cos(heading_yaw_dt));
    p.theta += yaw_rate * delta_t;
  }
}

void ParticleFilter::Predict(double delta_t, const double *std_pos, double velocity) {
  for (auto &p : particles_) {
    const double v_dt = velocity * delta_t;

		p.x += v_dt * cos(p.theta);
		p.y += v_dt * sin(p.theta);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
																	 std::vector<LandmarkObs> &observations,
																	 const Map &map) {
  ExecTime timing;

	for (auto &p : particles_) {
		p.associations.clear();

		for (LandmarkObs &obs : observations) {
			Association landmark_assoc;
			landmark_assoc.obs_in_ws = p.TransformToWorldSpace(obs);
			landmark_assoc.landmark_id = 0;

			// find distances to each land mark
			const auto count = map.landmark_list.size();
      Point landmark_pt;
      double min_distance = INT_MAX;

      for (const auto lm : map.landmark_list) {
				const double distance = landmark_assoc.obs_in_ws.GetDistance(lm.x_f, lm.y_f);

				if (distance < sensor_range && distance < min_distance) {
					min_distance = distance;
					landmark_assoc.landmark_id = lm.id_i;
          landmark_pt.x = lm.x_f;
          landmark_pt.y = lm.y_f;
				}
			}

      assert(landmark_assoc.landmark_id > 0);
			p.associations.push_back(landmark_assoc);

			const double obs_weight = multivariate_guassian(landmark_assoc.obs_in_ws, landmark_pt,
                                                      std_landmark[0], std_landmark[1]);

//			cout << "Observation Weight: " << std::scientific << obs_weight << endl;
			p.weight *=  obs_weight;
		}

//		cout << "Final Weight: " << std::scientific << p.weight << fixed << endl;
	}

//  cout << "Time taken weight: " << fixed << timing.End() << endl;
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

  for (auto &p : particles_)
    p.weight = 1;
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


double ParticleFilter::multivariate_guassian(const Point &obs_in_ws, const Point &landmark_pt, double std_x, double std_y)
{
	// calculate multivariate Gaussian
	// e = ((x - mu_x) ** 2 / (2.0 * std_x ** 2)) + ((y - mu_y) ** 2 / (2.0 * std_y ** 2))
	// e = ((diff_x ** 2) / (2.0 * std_x ** 2)) + ((diff_y ** 2) / (2.0 * std_y ** 2))

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
