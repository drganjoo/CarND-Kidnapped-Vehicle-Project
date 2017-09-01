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
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	random_device rd;
	mt19937 gen(rd());

	for (auto &p : particles_) {
		const double vel_per_rate = velocity / yaw_rate;
		const double heading_yaw_dt = p.theta + yaw_rate * delta_t;

		p.x += vel_per_rate * (sin(heading_yaw_dt) - sin(p.theta));
		p.y += vel_per_rate * (cos(p.theta) - cos(heading_yaw_dt));
		p.theta += yaw_rate * delta_t;

		normal_distribution<double> dist_x(p.x, std_pos[0]);
		normal_distribution<double> dist_y(p.y, std_pos[1]);
		normal_distribution<double> dist_theta(p.theta, std_pos[2]);

		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);

		// assuming angle is in radians not degrees, we need to make sure it does not
		// cross 2 * pi
		assert(p.theta >= 0 && p.theta <= M_PI * 2);
		p.theta = normalize_angle_rad(p.theta);
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> &observations,
	const Map &map) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles_ are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  //  x_changed = x_particle + x_obs * cos(heading_particle) - y_obs * sin(heading_particle)
  //  y_changed = y_particle + x_obs * sin(heading_particle) + y_obs * cos(heading_particle)
	for (auto &p : particles_) {
		p.associations.clear();

		for (LandmarkObs &obs : observations) {
			Association landmark_assoc;
			landmark_assoc.pt_in_ws = p.TransformToWorldSpace(obs);
			landmark_assoc.index = -1;

			// find distances to each land mark
			const auto count = map.landmark_list.size();
			double min_distance = INT_MAX;

			for (unsigned int landmark_index = 0; landmark_index < count; landmark_index++) {
				// distance = sqrt((x2 - x1 + y2 - y1) ** 2)
				// since we are just going to use the distance for choosing the closest one
				// there is no need to do a sqrt on the distance calculated

				const auto &lm = map.landmark_list[landmark_index];
				const double distance = landmark_assoc.pt_in_ws.GetDistance(lm.x_f, lm.y_f);

				if (distance < sensor_range && distance < min_distance) {
					min_distance = distance;
					landmark_assoc.index = landmark_index;
				}
			}

			p.associations.push_back(landmark_assoc);

			assert(landmark_assoc.index >= 0);
			
            const double obs_weight = multivariate_guassian(landmark_assoc, map, std_landmark[0], std_landmark[1]);
            cout << "Observation Weight: " << std::scientific << obs_weight << endl;
            p.weight *=  obs_weight;
		}
        
        cout << "Final Weight: " << std::scientific << p.weight << endl;
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
	std::transform(best.associations.begin(), best.associations.end(), ostream_iterator<double>(ss, " "), [](const Association &a) { return a.index; });
	std::string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseX(const Particle &best) {
	return getVectorToString(best.associations, [](const Association &a) { return a.pt_in_ws.x; });
}

string ParticleFilter::getSenseY(const Particle &best) {
	return getVectorToString(best.associations, [](const Association &a) { return a.pt_in_ws.y; });
}


double ParticleFilter::multivariate_guassian(const Association &association, const Map &map, double std_x, double std_y)
{
	// calculate multivariate Gaussian
	// e = ((x - mu_x) ** 2 / (2.0 * std_x ** 2)) + ((y - mu_y) ** 2 / (2.0 * std_y ** 2))
	// e = ((diff_x ** 2) / (2.0 * std_x ** 2)) + ((diff_y ** 2) / (2.0 * std_y ** 2))

	const Map::single_landmark_s &landmark = map.landmark_list[association.index];
	double diff_x = association.pt_in_ws.x - landmark.x_f;
	double diff_y = association.pt_in_ws.y - landmark.y_f;

	diff_x *= diff_x;
	diff_y *= diff_y;

	const double var_x = std_x * std_x;
	const double var_y = std_y * std_y;

	const double exponent = diff_x / (2.0 * var_x) + diff_y / (2.0 * var_y);

	// gauss_norm = (1.0 / (2.0 * pi * std_x * std_y))
	const double gauss_norm = 1.0 / (2.0 * M_PI * std_x * std_y);
	return gauss_norm * exp(-exponent);
}
