/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"

struct Point
{
	double x;
	double y;

	Point() = default;
	
	Point(double init_x, double init_y) {
		x = init_x;
		y = init_y;
	}

	inline double GetDistance(const Point &other) {
		const double diff = x - other.x + y - other.y;
		const double distance = sqrt(diff * diff);
		return distance;
	}

	Point TranslateRotate(const Point &translate_by, double theta) {
		assert(theta >= 0 && theta <= 2 * M_PI);

		const double p_cos = cos(theta);
		const double p_sin = sin(theta);

		Point tr_point;
		tr_point.x = translate_by.x + this->x * p_cos - this->y * p_sin;
		tr_point.y = translate_by.y + this->x * p_sin + this->y * p_cos;

		return tr_point;
	}
};

struct Association
{
	Point pt;
	int landmark_index;
};

struct Particle {

	int id;
	Point location;
	double theta;
	double weight;
	std::vector<Association> associations;

	Particle(int init_id, double init_x, double init_y, double init_theta) {
		id = init_id;
		location.x = init_x;
		location.y = init_y;
		theta = init_theta;

		weight = 1;
	}

	inline void SetAngle(double theta) {
		this->theta = theta;
	}
};



class ParticleFilter {
	
private:
	const int NUM_PARTICLES = 5;
	bool is_initialized;
	
public:
	
	// Set of current particles_
	std::vector<Particle> particles_;

	// Constructor
	// @param num_particles_ Number of particles_
	ParticleFilter() : is_initialized(false) {}

	// Destructor
	~ParticleFilter() {}

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);

	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
	 *   standard deviation of bearing [rad]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[],
                     std::vector<LandmarkObs> &observations,
			               const Map &map_landmarks);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	/*
	 * Set a particles_ list of associations, along with the associations calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
	 */
	void SetAssociations(Particle *particle, const std::vector<int> &associations,
                           const std::vector<double> &sense_x,
                           const std::vector<double> &sense_y);
	
	std::string getAssociations(const Particle &best);
	std::string getSenseX(const Particle &best);
	std::string getSenseY(const Particle &best);

	/**
	 * initialized Returns whether particle filter is initialized yet or not.
	 */
	const bool initialized() const {
		return is_initialized;
	}

private:
    template <typename T>
    std::string getVectorToString(const std::vector<T> v){
      std::stringstream ss;
      std::copy(v.begin(), v.end(), std::ostream_iterator<T>(ss, " "));
      std::string s = ss.str();
      s = s.substr(0, s.length() - 1);  // get rid of the trailing space
      return s;
    }

    double multivariate_guassian(const Point &pt_in_ws,
                                 const Map::single_landmark_s &landmark, double std_x, double std_y);
};



#endif /* PARTICLE_FILTER_H_ */
