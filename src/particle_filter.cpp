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

#include "particle_filter.h"

#include "helper_functions.h"

using namespace std;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 100;

	double std_x, std_y, std_psi; // Standard deviations for x, y, and psi
	std_x = std[0];
	std_y = std[1];
	std_psi = std[2];


	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_psi(theta, std_psi);

	for (int i = 0; i < num_particles; ++i) {
		int id;
		double sample_x, sample_y, sample_psi,weight;
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_psi = dist_psi(gen);
		weight = 1.0;
		id = i;
		Particle particle;

		particle.id = id;
		particle.x = sample_x;
		particle.y = sample_y;
		particle.theta = sample_psi;
		particle.weight = weight;

		particles.push_back(particle);
		is_initialized =true;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	double std_x, std_y, std_psi; // Standard deviations for x, y, and psi
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_psi = std_pos[2];

	for(int i=0;i<num_particles;i++){

		if (fabs(yaw_rate) < 0.00001) {
			particles[i].x = particles[i].x + velocity*delta_t*(cos(particles[i].theta));// + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y = particles[i].y  + velocity*delta_t*(sin(particles[i].theta));//- cos(particles[i].theta + yaw_rate*delta_t));
 		}
		else{
			 double v_by_yaw = velocity/yaw_rate;
			 particles[i].x = particles[i].x + v_by_yaw*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			 particles[i].y = particles[i].y  + v_by_yaw*(cos(particles[i].theta)- cos(particles[i].theta + yaw_rate*delta_t));
		}

		particles[i].theta = particles[i].theta + yaw_rate*delta_t;

		normal_distribution<double> dist_x(particles[i].x, std_x);
		normal_distribution<double> dist_y(particles[i].y, std_y);
		normal_distribution<double> dist_psi(particles[i].theta, std_psi);
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_psi(gen);

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
		for(int i=0;i<observations.size();i++){
				LandmarkObs observed =  observations[i];
				double min_dist = numeric_limits<double>::max();
				int minId =  -1;

				for(int j=0;j<predicted.size();j++){
					LandmarkObs predLandmark =  predicted[j];
					double distance = dist(observed.x,observed.y,predLandmark.x,predLandmark.y);
					if(distance < min_dist){
							min_dist = distance;
							minId = predLandmark.id;
					}
				}
				observations[i].id = minId;
		}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	for(int i=0;i<num_particles;i++){

		double p_x = particles[i].x;
 		double p_y = particles[i].y;
 		double p_theta = particles[i].theta;

		vector<LandmarkObs> predictions;

    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      // get id and x,y coordinates
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;

      if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {
      	predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
      }
    }

    // create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
    vector<LandmarkObs> transformed_os;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      transformed_os.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
    }

		dataAssociation(predictions, transformed_os);

		particles[i].weight = 1.0;

		for (int j = 0; j < transformed_os.size(); j++) {

      double o_x, o_y, pr_x, pr_y;
      o_x = transformed_os[j].x;
      o_y = transformed_os[j].y;

      int associated_prediction = transformed_os[j].id;

      for (int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == associated_prediction) {
          pr_x = predictions[k].x;
          pr_y = predictions[k].y;
        }
      }

      // calculate weight for this observation with multivariate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(pr_x-o_x,2)/(2*pow(s_x, 2)) + (pow(pr_y-o_y,2)/(2*pow(s_y, 2))) ) );

      // product of this obersvation weight with total observations weight
      particles[i].weight *= obs_w;
    }

	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	int index = rand()%num_particles;
	double beta = 0.0;
	vector<double> weights;
  double maxWieght = particles[0].weight;
	for (int i = 0; i < num_particles; i++) {
		if(particles[i].weight > maxWieght){
			maxWieght = particles[i].weight;
		}
		weights.push_back(particles[i].weight);
	}

	uniform_real_distribution<double> unirealdist(0.0, maxWieght);
	vector<Particle> new_particles;

	for(int i=0;i<num_particles;i++){
		beta += unirealdist(gen) * 2.0;
		  while (beta > weights[index]) {
			beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
