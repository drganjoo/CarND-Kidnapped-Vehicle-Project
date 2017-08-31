/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

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
#include <cmath>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles_. Initialize all particles_ to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles_ = 5;

  default_random_engine eng;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (unsigned int i = 0; i < num_particles_; i++) {
    Particle p (i, dist_x(eng), dist_y(eng), dist_theta(eng));

    particles_.push_back(p);
    weights_.push_back(1);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

//  x_f = x + v / yaw_rate * (sin(heading + yaw_rate * dt) - sin(heading))
//  y_f = y + v / yaw_rate * (cos(heading) - cos(heading + yaw_rate * dt))
//  heading_f = heading + yaw_rate * dt
  default_random_engine eng;

  for (auto &p : particles_)  {
    const double vel_per_rate = velocity / yaw_rate;
    const double heading_yaw_dt = p.theta + yaw_rate * delta_t;

    p.x += vel_per_rate * (sin(heading_yaw_dt) - sin(p.theta));
    p.y += vel_per_rate * (cos(p.theta) - cos(heading_yaw_dt));
    p.theta += yaw_rate * delta_t;

    normal_distribution<double> dist_x(p.x, std_pos[0]);
    normal_distribution<double> dist_y(p.y, std_pos[1]);
    normal_distribution<double> dist_theta(p.theta, std_pos[2]);

    p.x += dist_x(eng);
    p.y += dist_y(eng);
    p.theta += dist_theta(eng);

    // assuming angle is in radians not degrees, we need to make sure it does not
    // cross 2 * pi
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
    p.sense_x.clear();
    p.sense_y.clear();

    const double p_cos = cos(p.theta);
    const double p_sin = sin(p.theta);

    for (LandmarkObs &obs : observations) {
      obs.id = -1;

      const double obs_x_in_ps = p.x + obs.x * p_cos - obs.y * p_sin;
      const double obs_y_in_ps = p.y + obs.y * p_sin + obs.y * p_cos;

      double min_distance = INT_MAX;

      // find distances to each land mark
      const auto count = map.landmark_list.size();
      for (unsigned int landmark_index = 0; landmark_index < count; landmark_index++) {
        // distance = sqrt((x2 - x1 + y2 - y1) ** 2)
        // since we are just going to use the distance for choosing the closest one
        // there is no need to do a sqrt on the distance calculated

        const auto &lm = map.landmark_list[landmark_index];
        const double diff = lm.x_f - obs_x_in_ps + lm.y_f - obs_y_in_ps;
        const double distance = diff * diff;

        if (distance < min_distance) {
          min_distance = distance;
          obs.id = landmark_index;
        }
      }

      assert(obs.id >= 0);
      p.associations.push_back(obs.id);
      p.sense_x.push_back(obs_x_in_ps);
      p.sense_y.push_back(obs_y_in_ps);

      p.weight *= multivariate_guassian(obs_x_in_ps, obs_y_in_ps,
                                        map.landmark_list[obs.id], std_landmark[0], std_landmark[1]);
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles_ with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::SetAssociations(Particle *particle, const std::vector<int> &associations,
                                     const std::vector<double> &sense_x,
                                     const std::vector<double> &sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle->associations.clear();
  particle->sense_x.clear();
  particle->sense_y.clear();

  particle->associations = associations;
  particle->sense_x = sense_x;
  particle->sense_y = sense_y;
}

string ParticleFilter::getAssociations(const Particle &best) {
  vector<int> v = best.associations;
  return getVectorToString(v);
//  stringstream ss;
//  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
//  string s = ss.str();
//  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
//  return s;
}

string ParticleFilter::getSenseX(const Particle &best) {
  vector<double> v = best.sense_x;
  return getVectorToString(v);
//  stringstream ss;
//  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
//  string s = ss.str();
//  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
//  return s;
}

string ParticleFilter::getSenseY(const Particle &best) {
  vector<double> v = best.sense_y;
  return getVectorToString(v);
//  stringstream ss;
//  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
//  string s = ss.str();
//  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
//  return s;
}


double ParticleFilter::multivariate_guassian(double obs_x_in_ps, double obs_y_in_ps,
                             const Map::single_landmark_s &landmark,
                             double std_x, double std_y)
{
  // calculate multivariate gaussian
  // e = ((x - mu_x) ** 2 / (2.0 * std_x ** 2)) + ((y - mu_y) ** 2 / (2.0 * std_y ** 2))
  // e = ((diff_x ** 2) / (2.0 * std_x ** 2)) + ((diff_y ** 2) / (2.0 * std_y ** 2))

  double diff_x = obs_x_in_ps - landmark.x_f;
  double diff_y = obs_y_in_ps - landmark.y_f;

  diff_x *= diff_x;
  diff_y *= diff_y;

  const double var_x = std_x * std_x;
  const double var_y = std_y * std_y;

  const double exponent = diff_x / (2.0 * var_x) + diff_y / (2.0 * var_y);

  // gauss_norm = (1.0 / (2.0 * pi * std_x * std_y))
  const double gauss_norm = 1.0 / (2.0 * M_PI * std_x * std_y);
  return gauss_norm * exp(-exponent);
}