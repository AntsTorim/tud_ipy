# -*- coding: utf-8 -*-
from random import randrange, random, uniform, shuffle, choice
import numpy as numpy
import scipy as scipy
from pandas import DataFrame
from kernel_fca_oo import ConceptChain, FCASystemDF
from scipy.stats import binom, uniform as uniform_dist, norm, gamma, expon, poisson, bernoulli
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import csv
import sys
from itertools import combinations

# fetch object value distribution. 1-st row is weight from 0 (y) and second row defines weight position (x).
# x = protsent
# y = kaal diagrammil
# Saab lugeda 
def read_data(file):
     file_data = []
     try:
          with open(file) as csvfile:
               row_type = 'weight'
               aspect_rows = []
               file = csv.reader(csvfile, delimiter=',')
               for row in file:
                    row_formatted = []
                    for x in row:
                         row_formatted.append(float(x))
                    if row_type == 'weight':
                         aspect_rows.append(row_formatted)
                         row_type = 'range'
                    elif row_type == 'range':
                         aspect_rows.append(row_formatted)
                         file_data.append(aspect_rows)
                         aspect_rows = []
                         row_type = 'weight'
     except OSError:
          print(file + ' input error.')
          sys.exit()
     return file_data
'''
def read_data_relations(file_relations):
     # fetch object to object relation values - currently used for tests not samples
     file_relations_data = []
     try:
          with open(file_relations) as csvfile:
               relation_rows = []
               file = csv.reader(csvfile, delimiter=',')
               for row in file:
                    row_formatted = []
                    for x in row:
                         row_formatted.append(float(x))
                    relation_rows.append(row_formatted)
     except OSError:
          print('Relations file error. Continues without test aspect relations.')
     return file_relations_data
'''
# get list combinations from list X with pairs of 2. Ignores same lsit positions. Used for test relations generation.
# siin annab miskipärast 0,0 kombinatsiooni
def create_subsets(obj_amount):
     subsets = []
     for i in range (0, obj_amount):
          for k in range (0, obj_amount):
               already_in_list = 0
               for m in range(0, len(subsets)):
                    if i == k:
                         already_in_list = 1
                    elif (subsets[m][0] == k and subsets[m][1] == i):
                         already_in_list = 1
               if already_in_list == 0:
                    subset = []
                    subset.append(i)
                    subset.append(k)
                    subsets.append(subset)
     subsets.pop(0)
     return subsets

# used for test relations generation. Relation is generated per object subset. Example: object id 1 and 2 relation percentage is 25%. if student passed test 1, then doing 2 gets 25% of created extra ELO.
def generate_obj_relations(max_relation, obj_amount):
     obj_pos = []
     for i in range (0, obj_amount):
          obj_pos.append(i)
     relations = create_subsets(obj_amount)
     for i in range (0, len(relations)):
          relations[i].append(uniform(0.0, max_relation))
     return relations

# fetch object to object relation values - currently used for tests not samples.
# defines used aspect percentage per object. created metadata for aspect ratio generation.
def read_aspect_percentiles(file_aspect_percentiles, amount):
     file_aspect_percentiles_data = []
     try:
          with open(file_aspect_percentiles) as csvfile:
               aspect_percentiles_rows = []
               file = csv.reader(csvfile, delimiter=',')
               for row in file:
                    file_aspect_percentiles_data = row
     except OSError:
          print('Aspect ratio percentile file error. Continues without test aspect relations.')
     length = len(file_aspect_percentiles_data)
     created_percentile = 0
     for i in range(0, length):
          file_aspect_percentiles_data[i] = float(file_aspect_percentiles_data[i])
          created_percentile = created_percentile + float(file_aspect_percentiles_data[i])
     disabled_amount = amount - length
     if disabled_amount > 0:
          disabled_percentile = (100 - created_percentile)/disabled_amount
          for i in range(0, disabled_amount):
               file_aspect_percentiles_data.append(float(disabled_percentile))
     return file_aspect_percentiles_data

# scales data to defined points count. Example: input 20 object X/Y points. scaled to 100 by default.
# spline_degree - hard coded. 1-3 value. defines how smooth is the scaling process for filling the gap between data points.
def scale_data(array_of_power_positions, to_scale = 100, spline_degree = 3):
     scaled_list = []
     power_val = numpy.array(array_of_power_positions[0])
     power_val_pos = numpy.array(array_of_power_positions[1])
     scales_positions = numpy.linspace(power_val_pos.min(), power_val_pos.max(), to_scale)
     spline = make_interp_spline(power_val_pos, power_val, spline_degree)
     power_level = spline(scales_positions)
     plt.plot(power_level)
     plt.show()
     scaled_list.append(power_level)
     scaled_list.append(scales_positions)
     return scaled_list

# generates proibabilities for each possible ELO outcome. ELO outcome is created per aspect ELO.
def generate_probabilities(array_of_power_positions):
     power_val =  numpy.array(array_of_power_positions[0]) # weight
     power_val_pos = numpy.array(array_of_power_positions[1]) #range
     min_power_lev = power_val.min()
     power_weights = []
     for power_lev in power_val:
          power_weights.append(power_lev/min_power_lev)
     power_weights_combined = 0
     for power_weight in power_weights:
          power_weights_combined = power_weights_combined + power_weight
     power_probabilities = []
     for power_weight in power_weights:
          power_probabilities.append(power_weight/power_weights_combined)
     probabilities_for_val = []
     for i in range(0, len(power_val)):
          obj = []
          obj.append(power_probabilities[i])
          obj.append(power_val_pos[i])
          probabilities_for_val.append(obj)
     print(probabilities_for_val)
     return probabilities_for_val

# main function for cycling object generation. WORKS IN MASTER DATA MODE.
def generate_synth_objects(probabilities_with_val, amount):
     objs = []
     for i in range(0, amount):
          objs.append(generate_synth_object(probabilities_with_val))
     return objs

# function to generate objects by probabilities per aspect.
def generate_synth_object(probabilities_with_val):
     obj = []
     for i in range(0, len(probabilities_with_val)):
          weight_pos = random()
          current_weight = 0.00
          for k in range(0, len(probabilities_with_val[i])):
               weight = current_weight + float(probabilities_with_val[i][k][0])
               if current_weight < weight_pos and weight_pos <= weight:
                  obj.append(probabilities_with_val[i][k][1])
                  break
               else:
                   current_weight = weight
     return obj

# generates aspect percentiles for tests. Maximum percentageof which the aspect can be used.
def generate_aspects_percentiles(Test_aspects_ratios, Objects_len):
     percentiles = []
     percentiles_total_predefined = 0.0
     for i in range(0, len(Test_aspects_ratios)):
          percentiles_total_predefined = percentiles_total_predefined + float(Test_aspects_ratios[i])
     if percentiles_total_predefined > 100.00:
          print('Percentiles over 100%. Correct data before continuing.')
          sys.exit()
     for i in range(0, Objects_len):
          obj = []
          for k in range(0, len(Test_aspects_ratios)):
               obj.append(uniform(0.0, Test_aspects_ratios[k]))
          percentiles.append(obj)
     for i in range(0, len(percentiles)):
          percentiles_check = 0.0
          for k in range(0, len(percentiles[i])):
               percentiles_check = percentiles_check + float(percentiles[i][k])
          if percentiles_check < 1.0:
               percentile_multiplier = 1.0/percentiles_check
               for k in range(0, len(percentiles[i])):
                    percentiles[i][k] = percentiles[i][k]*percentile_multiplier
     print(percentiles)
     return percentiles

def calc_test_pass(elo):
     return int(elo > uniform(0.0, 1.0))

def calc_elo_probability(elo_val_1, elo_val_2):
     elo_1 = 10**(elo_val_1 / 400)
     elo_2 = 10**(elo_val_2 / 400)
     return elo_1 / (elo_1 + elo_2)

# calculates object total competencies across aspects
def calc_object_elo(object_aspects, aspect_ratios):
     total_elo = 0.0
     for i in range(0, len(object_aspects)):
          total_elo = total_elo + (object_aspects[i]*aspect_ratios[i])
     return total_elo

# each aspect shows the complexity level. Ratio per aspect defines percentage of test.
# hetkel loogika arvestab, et aspect ratio kehtib mõlemale objektile.
def compare_elo(sample_aspects, test_aspects, test_aspect_ratios):
     sample_elo = calc_object_elo(sample_aspects, test_aspect_ratios)
     test_elo = calc_object_elo(test_aspects, test_aspect_ratios)
     elo_probability = calc_elo_probability(sample_elo, test_elo)
     result = []
     result.append(calc_test_pass(elo_probability))
     result.append(elo_probability)
     result.append(sample_elo)
     result.append(test_elo)
     return result

# edasiarendus täpsuse mõistes rekursiivselt. Läheneb hetkel loogikaga, et testis on juba ELO-s peidetud eelnevate teadmine. Rekursioon tagab ainult täpsust. Nt kui kuni 0.25 ühest testist ja kuni 0.25 teisest testist, siis reaalne ELO teisest testist võib olla circa ~3 ELO punkti keskmiselt 
# trackib iga elo aspectis eraldi
# experience confidence over time bayesian interference https://towardsdatascience.com/estimating-probabilities-with-bayesian-modeling-in-python-7144be007815
# passed_tests_exp_elo = [[x x x], [x x x]]
# kui võrd täppi läheb ja selle alusel kontrollida. Alternatiiv konseptiahelatele.
# bayesian interference not implemented, https://en.wikipedia.org/wiki/Posterior_probability https://en.wikipedia.org/wiki/Bayes%27_theorem. Easier version created. as this case lacks 2 different types of probabilities (only chance of winning, but not chance of alternative connected probability)
# interference utilizes probability of winning currently and probability that the user won in history
# this function tries to implement knowledge space logic
# interference_weight = me ei tea hinnet, seega läbimise puhul on see 0-50%. Confidence level  hindab potentsiaalset soorituste taset. Piisab high-level tasemel keskmsie vaatamisest.
def calc_experienced_aspects(current_test_no, object_aspects, test_relations, passed_tests, passed_tests_exp_elo, current_test_elo, interference = 1, interference_weight = 0.5):
     current_aspects = object_aspects
     confidence_lev = 1
     if interference == 1:
          passed_tests_amount = 0
          for i in range(0, len(passed_tests)):
               if passed_tests[i] == 1:
                    passed_tests_amount = passed_tests_amount + 1
          if len(passed_tests) > 0:
               confidence_lev =  confidence_lev*(1-interference_weight) + interference_weight*(passed_tests_amount/len(passed_tests))
          else:
               confidence_lev = 1
     for i in range(0, current_test_no):
          if passed_tests[i] == 1:
               for k in range(0, len(test_relations)):
                    if test_relations[k][0] == i and test_relations[k][1] == current_test_no:
                         for m in range(0, len(current_aspects)):
                              current_aspects[m] = (current_aspects[m] + (float(test_relations[k][2])*passed_tests_exp_elo[i][m]))*confidence_lev
     return current_aspects

#The long-term average for teams is 1500 and values generally range from 1200 to 1800.
#elo regression to be implemented for test run? technically dynamic elo already provides historical data and bayes interf. utilizes it https://en.wikipedia.org/wiki/Logistic_function
#auto_corr_rand = g value https://github.com/ddm7018/Elo
#lower limit is 0, never is lowered
#react_speed - K value of ELO changing over results
def calc_added_ELO(win_prob, test_ratios, pass_truth, react_speed = 20, auto_corr_rand = 1):
     result_array = []
     for i in range(0, len(test_ratios)):
          if pass_truth == 1:
               result_array.append((react_speed*auto_corr_rand)*(1 - win_prob))
          else:
               result_array.append(float(0))
     return result_array

# calculates object total competencies for aspects
def generate_synth_data(Sample_objects, Test_objects, Test_aspects_ratios, dynamic_elo = 1, fail_condition = 0, Test_relations = [], total_fail_check_cycle = 6, cycle_allowed_fail_no = 2):
     synth_data = []
     validation_elo_sequence = []
     validation_elo_probability = []
     for sample in range(0, len(Sample_objects)):
          sample_elo_results = []
          sample_elo_result_elos = [] # hoiab arraysid per
          validation_sample_elo_sequence = []
          validation_sample_probability = []
          sample_has_failed = 0
          for test in range(0, len(Test_objects)):
               if (test % total_fail_check_cycle == 0) and fail_condition == 1:
                    allowed_fails = (test/total_fail_check_cycle)*cycle_allowed_fail_no
                    sample_elo_results_sum = 0
                    for e in range(0, len(sample_elo_results)):
                        sample_elo_results_sum = sample_elo_results_sum + sample_elo_results[e]
                    if (sample_elo_results_sum + allowed_fails) < len(sample_elo_results):
                         sample_has_failed = 1
               if sample_has_failed == 0:
                    sample_obj = Sample_objects[sample]
                    if len(Test_relations) > 0 and dynamic_elo == 1:
                         sample_obj = calc_experienced_aspects(test, sample_obj, Test_relations, sample_elo_results, sample_elo_result_elos, Test_objects[test])
                    result = compare_elo(sample_obj, Test_objects[test], Test_aspects_ratios[test])
                    sample_elo_results.append(result[0])
                    sample_elo_result_elos.append(calc_added_ELO(result[1], Test_aspects_ratios[test], result[0]))
                    val_obj = []
                    val_obj.append(result[1])
                    val_obj.append(result[2])
                    val_obj.append(result[3])
                    validation_sample_probability.append(val_obj)
                    validation_sample_elo_sequence.append(sample_obj)
               else:
                    sample_elo_results.append(int(0))
          validation_elo_sequence.append(validation_sample_elo_sequence)
          validation_elo_probability.append(validation_sample_probability)
          synth_data.append(sample_elo_results)
     return numpy.array(synth_data), numpy.array(validation_elo_sequence), numpy.array(validation_elo_probability)

def generate_statistics():
     statistics_data = []
     
     return statistics_data

def generate_fca_concepts():
     fca_concepts_data = []
     
     return fca_concepts_data

def generate_data_by_master(samples_file, generate_samples_amount, tests_file, generate_tests_amount, test_aspect_ratios, sort_tests_asc = 1, dynamic_elo = 1, fail_condition = 0, interference = 1, scaling_level = 1000):
# Generate master data module
     # read master knowledge
     Sample_aspects = read_data(samples_file)
     Tests_aspects = read_data(tests_file)

     # smooth out & scale expert data
     Sample_aspects_scaled = []
     for aspect in Sample_aspects:
           Sample_aspects_scaled.append(scale_data(aspect, scaling_level))
     Tests_aspects_scaled = []
     for aspect in Tests_aspects:
           Tests_aspects_scaled.append(scale_data(aspect, scaling_level))

     # generate aspect powers for objects
     Sample_aspect_powers = []
     for aspect in Sample_aspects_scaled:
           Sample_aspect_powers.append(generate_probabilities(aspect))
     Tests_aspect_powers = []
     for aspect in Tests_aspects_scaled:
           Tests_aspect_powers.append(generate_probabilities(aspect))
           
# Generate synthetic ELO objects module
     # generate objects
     Sample_objects = generate_synth_objects(Sample_aspect_powers, generate_samples_amount)
     Test_objects = generate_synth_objects(Tests_aspect_powers, generate_tests_amount)

# Generate Test aspect ratios module - if not set, is even
     if len(test_aspect_ratios) > 0:
          Test_aspects_ratios_list = read_aspect_percentiles(test_aspect_ratios, len(Tests_aspects))
     else:
          Test_aspects_ratios_list = []
     Test_aspects_percentiles = generate_aspects_percentiles(Test_aspects_ratios_list, len(Test_objects))

# Organize Test objects module
     if sort_tests_asc == 1:
          Test_objects = sorted(Test_objects, key=sum)

# Generate Test relations module - randomized, is not sorted with test objects
     #Test_relations = read_data_relations('E:\\test_relations.csv') deprecated
     if dynamic_elo == 1:
          Test_relations = generate_obj_relations(0.25, len(Test_objects))
     else:
          Test_relations = []

# Generate synthetic ELO results module v1
     Synth_data = generate_synth_data(Sample_objects, Test_objects, Test_aspects_percentiles, dynamic_elo = dynamic_elo, fail_condition = fail_condition, Test_relations = Test_relations, total_fail_check_cycle = 6, cycle_allowed_fail_no = 2)
     return Synth_data

# Function generates by given input and distribution type student/test objects. USER DOES NOT HAVE TO INCLUDE THEIR OWN DISTRIBUTION FILES.
# scikit implementation. Exceptions: 1) random - takes random ELO for aspect from range. 2) flat - base elo for every aspect, 3) rising_steady_aspect - within lowest to highest ELO with noise for aspects (example: 1) 1400, 2) 1500, 3) 1600)
# rising_steady_object - rising steady logic per object
# object_aspects - obj. aspects total count.
# object_noise_elo - defines range from base elo center.
# scramble_aspects - scrambles randomly created object aspects / random order for generated aspects (aspect is defined by index location. Random value scrambles ELo between aspects).
# scramble_objects - useful for tests. scrambles tests in list
# tilt_pos_elo - used for scikit explicit functions such as binomal, gamma, poisson, bernoulli distributions
# reverse_aspects - useful to reverse generated lsit of aspects
def generate_objects_by_function(object_amount, object_aspects, object_base_elo = 1500, object_noise_elo = 300, distribution_type = 'random', scramble_aspects = 1, scramble_objects = 1,
				 tilt_pos_elo = 1350, reverse_aspects = 0):
     objects = []
     if distribution_type == 'random':
          max_elo = object_base_elo + object_noise_elo
          min_elo = object_base_elo - object_noise_elo
          for i in range(0, object_amount):
               obj = []
               for k in range(0, object_aspects):
                    obj.append(randrange(min_elo, max_elo))
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)
     elif distribution_type == 'flat':
          for i in range(0, object_amount):
               obj = []
               for k in range(0, object_aspects):
                    obj.append(object_base_elo)
               objects.append(obj)
     elif distribution_type == 'rising_steady_aspect':
          rising_step = (object_noise_elo*2)/object_aspects
          for i in range(0, object_amount):
               obj = []
               for k in range(0, object_aspects):
                    obj.append((object_base_elo-object_noise_elo) + (i*rising_step))
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)
     elif distribution_type == 'rising_steady_object':
          rising_step = (object_noise_elo*2)/object_amount
          for i in range(0, object_amount):
               min_elo_obj = object_base_elo - object_noise_elo
               max_elo_obj = min_elo_obj + rising_step*(i)
               obj = []
               for k in range(0, object_aspects):
                    obj.append(max_elo_obj)
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)
     elif distribution_type == 'rising_steady_object_rand':
          rising_step = (object_noise_elo*2)/object_amount
          for i in range(0, object_amount):
               min_elo_obj = object_base_elo - object_noise_elo
               min_elo_obj = min_elo_obj + rising_step*(i)
               max_elo_obj = min_elo_obj + rising_step
               obj = []
               for k in range(0, object_aspects):
                    obj.append(randrange(min_elo_obj, max_elo_obj))
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)
     elif distribution_type == 'lowering_steady_aspect':
          lowering_step = (object_noise_elo*2)/object_aspects
          for i in range(0, object_amount):
               obj = []
               for k in range(0, object_aspects):
                    obj.append((object_base_elo+object_noise_elo) - (i*lowering_step))
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)
     elif distribution_type == 'lowering_steady_object':
          rising_step = (object_noise_elo*2)/object_amount
          for i in range(0, object_amount):
               min_elo_obj = object_base_elo + object_noise_elo
               max_elo_obj = min_elo_obj - rising_step*(i)
               obj = []
               for k in range(0, object_aspects):
                    obj.append(max_elo_obj)
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)
     elif distribution_type == 'lowering_steady_object_rand':
          rising_step = (object_noise_elo*2)/object_amount
          for i in range(0, object_amount):
               min_elo_obj = object_base_elo + object_noise_elo
               min_elo_obj = min_elo_obj - rising_step*(i)
               max_elo_obj = min_elo_obj - rising_step
               obj = []
               for k in range(0, object_aspects):
                    obj.append(randrange(min_elo_obj, max_elo_obj))
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)
     elif distribution_type == 'uniform':
          min_elo = object_base_elo - object_noise_elo
          for i in range(0, object_amount):
               obj = []
               for k in range(0, object_aspects):
                    dist = uniform_dist.rvs(size=2000, loc = min_elo, scale=(object_noise_elo*2))
                    obj.append(choice(dist))
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)
     elif distribution_type == 'normal':
          min_elo = object_base_elo - object_noise_elo
          for i in range(0, object_amount):
               obj = []
               for k in range(0, object_aspects):
                    dist = norm.rvs(size=2000,loc=min_elo,scale=(object_noise_elo*2))
                    obj.append(choice(dist))
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)
     elif distribution_type == 'binomal':
          min_elo = object_base_elo - object_noise_elo
          binom_pos = (tilt_pos_elo-min_elo)/(object_noise_elo*2)
          for i in range(0, object_amount):
               obj = []
               for k in range(0, object_aspects):
                    dist = binom.rvs(n=object_noise_elo*2,p=binom_pos,size=2000)
                    obj.append(min_elo + choice(dist))
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)   
     elif distribution_type == 'gamma':
          min_elo = object_base_elo - object_noise_elo
          for i in range(0, object_amount):
               obj = []
               elo_range = []
               step = (object_noise_elo*2)/2000
               for k in range(0, object_aspects):
                    elo_range.append(min_elo+k*step)
               for k in range(0, object_aspects):
                    dist = gamma.rvs(a=tilt_pos_elo, size=2000)
                    selected_index = 0
                    while selected_index == 0:
                         rand_loc = randrange(0, len(dist))
                         if dist[rand_loc] == 1:
                              selected_index = rand_loc
                    obj.append(elo_range[rand_loc])
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)
     elif distribution_type == 'poisson':
          min_elo = object_base_elo - object_noise_elo
          for i in range(0, object_amount):
               obj = []
               for k in range(0, object_aspects):
                    dist = poisson.rvs(mu=tilt_pos_elo, size=2000)
                    obj.append(choice(dist))
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)          
     elif distribution_type == 'exponential':
          min_elo = object_base_elo - object_noise_elo
          for i in range(0, object_amount):
               obj = []
               for k in range(0, object_aspects):
                    dist = expon.rvs(size=2000,loc=min_elo,scale=(object_noise_elo*2))
                    obj.append(choice(dist))
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)
     elif distribution_type == 'bernoulli':
          min_elo = object_base_elo - object_noise_elo
          bernoulli_pos = (tilt_pos_elo-min_elo)/(object_noise_elo*2)
          for i in range(0, object_amount):
               obj = []
               elo_range = []
               step = (object_noise_elo*2)/2000
               for k in range(0, object_aspects):
                    elo_range.append(min_elo+k*step)
               for k in range(0, object_aspects):
                    dist = bernoulli.rvs(size=2000,p=bernoulli_pos)
                    selected_index = 0
                    while selected_index == 0:
                         rand_loc = randrange(0, len(dist))
                         if dist[rand_loc] == 1:
                              selected_index = rand_loc
                    obj.append(elo_range[rand_loc])
               if scramble_aspects == 1:
                    shuffle(obj)
               if reverse_aspects == 1:
                    obj.reverse()
               objects.append(obj)
     if scramble_objects == 1:
          shuffle(objects)
     return objects


# generate_samples_amount - students amount (row objects)
# generate_tests_amount - tests amount (column objects)
# aspects_amount - each object different aspects amount. I.e. math knowledge, physics knowledge, physical capability, IQ level etc. All row and column objects have same amount of aspects for comparison.
# test_aspect_ratios - defines list of aspect ratios for column objects / tests. If not set, all aspects will be evenly distributed. Aspects can be from 0%-defined%.
# sort_tests_asc - boolean value. Defines if the tests should be sorted from easiest to hardest.
# dynamic_elo - boolean value. turns on/off dynamic elo for students/rows. Rows gather experience from passing tests. Uses test relations.
# fail_condition - boolean value. Introduces failing condition. defined in code - cyclle which checks every 6 tests (hardcode changeable.) and maximum fail per cycle 2 (accumulates per cycle). Proovib ttü näidet rakendada, et 22.5 EAP peabo lema 30eap läbitud per nominaal semester.
# interference - boolean value. defines if defined amount (defines 0.5, in function) of gathered experience is based on pass/fail rate of statistically passed/failed tests. for example. if students passed test and rate is 0.5, but in past only half of the tests have been passed, then the confidence level of gathered experiewnce is in total 0.5 + 0.5(0.5)
# scaling_level - Used for manual input. Scales input values (x/y) to specified scale. If input is 20, then generated is 1000 points from 20.
# sample_base_elo - base average elo used for generation. For sample.
# sample_noise_elo - base elo maximum deviation in positive/negative. Defines maximum/minimum available elo for generation. For sample.
# scramble_aspects - scrambles object aspects (aspect lsit is randomly sorted) while generating objects.
# sample_tilt_pos_elo - when creating distribtion (binomal, gamma etc.) is used to define base location on min-max elo range. For sample.
# test_tilt_pos_elo - when creating distribtion (binomal, gamma etc.) is used to define base location on min-max elo range. For test.
# reverse_aspects - reverses object aspects.
# scramble_objects - scrambles objects in list. Effective for tests.
# test_base_elo - base average elo used for generation. For test.
# test_noise_elo - base elo maximum deviation in positive/negative. Defines maximum/minimum available elo for generation. For test.
# test_aspect_ratios_random - boolean value. (100% divided by aspects amount per aspect if not activated) Model assigns random test aspect ratios. Only if manual aspects ratio is not provided.
# sample_distribution_type - enum value.
# test_distribution_type - enum value.
# distribution types: random, flat, rising_steady_aspect, rising_steady_object, rising_steady_object_rand, lowering_steady_aspect, lowering_steady_object, lowering_steady_object_rand,
# distribution types by functions: uniform, bernoulli, exponential, poisson, gamma, binomal, normal. More from scikit modules.

def generate_data_by_input(generate_samples_amount, generate_tests_amount, aspects_amount, test_aspect_ratios = '', sort_tests_asc = 0, dynamic_elo = 0,
                           fail_condition_samples = 0, interference = 0, sample_base_elo = 1500, sample_noise_elo = 300, test_base_elo = 1500, test_noise_elo = 300,
                           sample_distribution_type = 'random', test_distribution_type = 'random', scramble_aspects = 0, scramble_objects = 0,
                           sample_tilt_pos_elo = 1350, test_tilt_pos_elo = 1350, reverse_aspects = 0, test_aspect_ratios_random = 0):
# Generate synthetic ELO objects module
     # generate objects
     Sample_objects = generate_objects_by_function(generate_samples_amount, aspects_amount, sample_base_elo, sample_noise_elo, sample_distribution_type,
                                                   scramble_aspects, scramble_objects, sample_tilt_pos_elo, reverse_aspects)
     Test_objects = generate_objects_by_function(generate_tests_amount, aspects_amount, test_base_elo, test_noise_elo, test_distribution_type, scramble_aspects,
                                                 scramble_objects, test_tilt_pos_elo, reverse_aspects)

# Generate Test aspect ratios module. All values provide aspect potential ratio from 0 - aspect max ratio.
     if len(test_aspect_ratios) > 0:
          Test_aspects_ratios_list = read_aspect_percentiles(test_aspect_ratios, len(Tests_aspects))
          Test_aspects_percentiles = generate_aspects_percentiles(Test_aspects_ratios_list, len(Test_objects))
     elif test_aspect_ratios_random == 1:
          aspects_len = len(Sample_objects[0])
          Test_aspects_percentiles = numpy.random.dirichlet(numpy.ones(aspects_len),size=len(Test_objects))
     else:
          Test_aspects_ratios_list = []
          aspects_len = len(Sample_objects[0])
          for i in range(0, aspects_len):
               Test_aspects_ratios_list.append(float(100/aspects_len)/100)

          print(Test_aspects_ratios_list)
          Test_aspects_percentiles = generate_aspects_percentiles(Test_aspects_ratios_list, len(Test_objects))

# Organize Test objects module
     if sort_tests_asc == 1:
          Test_objects = sorted(Test_objects, key=sum)

# Generate Test relations module - randomized, is not sorted with test objects
     #Test_relations = read_data_relations('E:\\test_relations.csv')
     if dynamic_elo == 1:
          Test_relations = generate_obj_relations(0.25, len(Test_objects))
     else:
          Test_relations = []

# Generate synthetic ELO results module v1
     Synth_data = generate_synth_data(Sample_objects, Test_objects, Test_aspects_percentiles, dynamic_elo = dynamic_elo, fail_condition = fail_condition_samples, Test_relations = Test_relations, total_fail_check_cycle = 6, cycle_allowed_fail_no = 2)
     return Synth_data

if __name__ == "__main__":
     #res1, val1_elo, val1_elo_prob  = generate_data_by_master(samples_file = 'E:\\stud_data.csv', generate_samples_amount = 20, tests_file = 'E:\\test_data.csv', generate_tests_amount = 20, test_aspect_ratios = 'E:\\aspect_ratios.csv', sort_tests_asc = 1, dynamic_elo = 1, fail_condition = 1, interference = 1, scaling_level = 1000)
     #print(res1)
     #print('per student case, per comparison, list [sample aspect elos, test aspect elos]')
     #print(val1_elo)
     #print('per student case, per comparison, list [win probability, sample total elo, test total elo]')
     #print(val1_elo_prob)
     #print('-------------------------Alternative-------------------------------')
     res2, val2_elo, val2_elo_prob = generate_data_by_input(generate_samples_amount = 20, generate_tests_amount = 20, aspects_amount = 5, sort_tests_asc = 1,
                                                            dynamic_elo = 1, fail_condition_samples = 0, interference = 1, sample_base_elo = 2000, sample_noise_elo = 300, test_base_elo = 1500,
                                                            test_noise_elo = 300, sample_distribution_type = 'random', test_distribution_type = 'random',
                                                            scramble_aspects = 1, sample_tilt_pos_elo = 1350, test_tilt_pos_elo = 1850, test_aspect_ratios_random = 1)
     print(res2)
     #print('per student case, per comparison, list [sample elo, test elo]')
     #print(val2_elo)
    # print('per student case, per comparison, probability number')
    # print(val2_elo_prob)
     res3, val3_elo, val3_elo_prob = generate_data_by_input(generate_samples_amount = 20, generate_tests_amount = 20, aspects_amount = 8)
     print(res3)
