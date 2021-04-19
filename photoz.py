#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:02:58 2020

@author: Xavier Jimenez
"""

#------------------------------------------------------------------#
# # # # # Imports # # # # #
#------------------------------------------------------------------#

import numpy as np
import os
import shutil
import glob
import pandas as pd
import importlib

from joblib import Parallel, delayed
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings('ignore')
from functions import *

#------------------------------------------------------------------#
# # # # # Create catalog # # # # #
#------------------------------------------------------------------#

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nodes", required=False, type=int, nargs="?", const=1)
    parser.add_argument("-s", "--survey", required=False, type=str, nargs="?", const='test')
    parser.add_argument("-c", "--clean", required=False, type=bool, nargs="?", const=False)
    parser.add_argument("-m", "--make", required=False, type=bool, nargs="?", const=False)
    parser.add_argument("-j", "--join", required=False, type=bool, nargs="?", const=False)
    parser.add_argument("-g", "--generate_plots", required=False, type=bool, nargs="?", const=False)
    parser.add_argument("-p", "--preprocess", required=False, type=str, nargs="?", const=None)    
    parser.add_argument("-l", "--learning", required=False, type=bool, nargs="?", const=False)
    parser.add_argument("-o", "--optimize", required=False, type=str, nargs="?", const=None)
    parser.add_argument("-a", "--algorithm", required=False, type=str, nargs="?", const='RF')
    parser.add_argument("-i", "--input", required=False, type=str)

    args = parser.parse_args()

#------------------------------------------------------------------#
# # # # # PS3PI # # # # #
#------------------------------------------------------------------#

    path = os.getcwd() + '/'
    if args.input is None:
        import params
    else:
        params = importlib.import_module(args.input)
    
    if args.nodes is None:
        args.nodes = 1

    if args.algorithm is None:
        args.algorithm = 'RF'

    if args.survey is None:
        args.survey = 'test'

    if args.survey == 'test':
        print('Modules loaded properly')

    if args.preprocess is None:
        args.preprocess = 'drop'

    elif args.survey == 'ps3pi_cfis' or args.survey == 'unions':
        
        bands = params.bands
        output_path = params.output_path
        output_name = params.output_name
        temp_path = params.temp_path

        #------------------------------------------------------------------#
        # # # # # CLEAN # # # # #
        #------------------------------------------------------------------#

        if args.clean == True:
            GenFiles = GenerateFiles(args.survey, bands, temp_path, output_name, output_path)
            GenFiles.clean_temp_directories()
            GenFiles.make_directories()

        #------------------------------------------------------------------#
        # # # # # MAKE INDIVIDUAL TILE CATALOGS # # # # #
        #------------------------------------------------------------------#

        if args.make == True:
            spectral_path = params.spectral_path
            spectral_names = params.spectral_names
            path_to_tile_run = params.path_to_tile_run
            spectral_surveys = params.spectral_surveys
            vignet = params.vignet


            cat = MakeCatalogs(args.survey, bands, temp_path, output_name, output_path)

            for i in range(len(spectral_names)):
                cat.make_survey_catalog(spectral_path, spectral_names[i])
                if params.input_path == None:
                    out_dir = os.listdir(path_to_tile_run + args.survey + '/%s/output/'%(spectral_surveys[i]))[-1]
                    input_path = path_to_tile_run + args.survey + '/%s/output/%s/paste_cat_runner/output/'%(spectral_surveys[i], out_dir)
                else:
                    input_path = params.input_path
                paste_dir = os.listdir(input_path)
                Parallel(n_jobs=args.nodes)(delayed(cat.make_catalog)(p, paste_dir, input_path, spectral_names[i], vignet=vignet) for p in tqdm(range(len(paste_dir))))


        #------------------------------------------------------------------#
        # # # # # JOIN INDIVIDUAL TILE CATALOGS # # # # #
        #------------------------------------------------------------------#

        if args.join == True:
            vignet = params.vignet
            cat = MakeCatalogs(args.survey, bands, temp_path, output_name, output_path)
            cat.merge_catalogs(vignet=vignet)

        #------------------------------------------------------------------#
        # # # # # SAVE FIGURES # # # # #
        #------------------------------------------------------------------#

        if args.generate_plots == True:
            spectral_names = params.spectral_names
            GenPlot = GeneratePlots(args.survey, bands, temp_path, output_name=output_name, spectral_names=spectral_names, output_path=output_path)
            GenPlot.plot_matched_z_spec_hist()
            GenPlot.plot_unmatched_z_spec_hist()

        #------------------------------------------------------------------#
        # # # # # MACHINE LEARNING ALGORITHMS # # # # #
        #------------------------------------------------------------------#

        if args.learning == True:

            GenFiles = GenerateFiles(args.survey, bands, path, output_name, output_path=output_path)
            GenFiles.make_directories(output=True)

            path_to_csv = params.path_to_csv
            spectral_names = params.spectral_names
            weights = params.weights
            cv = params.cv
            max_evals = params.max_evals
            feature_engineering = params.feature_engineering
            feature_importance = params.feature_importance
            plot = params.plot
            
            if path_to_csv is None:
                if args.survey == 'ps3pi_cfis':
                    path_to_csv = output_path + 'output/' + args.survey  + '/' + output_name + '/files/' + output_name + '.csv'
                    ML = LearningAlgorithms(survey = args.survey, bands = bands, path_to_csv = path_to_csv, output_name = output_name, output_path=output_path, cv=cv, preprocessing=args.preprocess, n_jobs=args.nodes)
                    df, df_unmatched = ML.merge_cfis_r_cfht_u_medium_deep_i_g_z()
                    if feature_engineering == True:
                        # df_list = ML.feature_engineering(df, bands=['r', 'u', 'i', 'z', 'g'])
                        df_list = ML.feature_engineering(df, bands=['r', 'u', 'i', 'z', 'g'], color_order=['i', 'g' , 'r', 'z', 'u'])
                    else:
                        df_list = [df]
                    # print(df.head(10))
                    if plot == True:
                        ML.plot_corrmat(df)
                        GenPlot = GeneratePlots(args.survey, bands, temp_path, output_name=output_name, output_path=output_path, spectral_names=spectral_names)
                        # GenPlot.plot_mags(df, df_unmatched)

                elif args.survey == 'unions':
                    path_to_csv = output_path + 'output/' + args.survey  + '/' + output_name + '/files/' + output_name + '.csv'
                    ML = LearningAlgorithms(survey = args.survey, bands = bands, path_to_csv = path_to_csv, output_name = output_name, output_path=output_path, cv=cv, preprocessing=args.preprocess, n_jobs=args.nodes)
                    df = ML.dataframe()
                    df_unmatched = ML.unmatched_dataframe()
                    df = ML.gal_g()
                    if plot == True:
                        ML.plot_corrmat(df)
                        GenPlot = GeneratePlots(args.survey, bands, temp_path, output_name=output_name, output_path=output_path, spectral_names=spectral_names)
                        GenPlot.plot_mags(df, df_unmatched)
                else:
                    raise TypeError("--survey needs to be set to 'unions' or 'ps3pi_cfis', please specify the full path to your DataFrame")

            elif path_to_csv is not None:
                # MKDEBUG: Added the following cases
                if weights == True:
                    # Don't know what to do here. weights=True in LearningAlgorithms gives error
                    sample_weights = None
                elif type(weights) == str:
                    sample_weights = np.load(weights)
                else:
                    sample_weights = None
                ML = LearningAlgorithms(survey = args.survey, bands = bands, path_to_csv = path_to_csv, output_name = output_name, output_path=output_path, sample_weight=sample_weights, cv=cv, preprocessing=args.preprocess, n_jobs=args.nodes)
                df = ML.dataframe()
                df_list = [df]
                # ML.plot_corrmat(df)

            algs = {'RF': RandomForest, 'ANN': ArtificialNeuralNetwork, 'LASSO': LassoRegression, 'ENET': ElasticNetRegression,
                        'XGB':XGBoost, 'KRR':KernelRidgeRegression, 'SVR': SupportVectorRegression, 'LGB': LightGBM, 'GBR': GradientBoostingRegression}
         
            if args.algorithm == 'BEST':

                algs = {'RF': RandomForest, 'ANN': ArtificialNeuralNetwork, 'SVR': SupportVectorRegression, 'GBR': GradientBoostingRegression}

                best_score = 1
                best_alg = 'none'
                # alg_names = np.array(list(algs.items()))[:,1]
                if weights == True:
                    cat = MakeCatalogs(args.survey, bands, temp_path, output_name, output_path)
                    weights = cat.compute_weights(df, column = 'r')
                elif type(weights) == str:
                    weights = np.load(weights)
                else:
                    weights = None
                global_score = 1
                best_dict = pd.DataFrame(data={}, index=['score', 'score std'])
                y_pred_dict = {}
                y_test_dict = {}

                for alg_name in algs:
                    best_score= 1
                    alg = algs[alg_name]       
                
                    print('[Feature engineering]')
                    print('---------------------------------------------------------------')
                        
                    for df in df_list:
                        method = alg(survey = args.survey, bands = bands, output_name = output_name, temp_path=temp_path, dataframe=df, path_to_csv=None, validation_set=False, output_path=output_path, sample_weight=weights, cv=cv, preprocessing=args.preprocess, n_jobs=args.nodes)
                        score = method.score()
                        print(list(df.columns))
                        print('[preprocess] %s'%score[4])
                        print('[%s '%alg_name +'score] {:.3f} ± {:.3f}'.format(score[5], score[6]))
                        if score[5] < best_score:
                            print('[NEW BEST]')
                            print("%s: "%alg_name + "Sigma: {:.3f} ± {:.4f}, outlier rate: {:.3f} ± {:.3f} % ".format(score[0], score[1], score[2]*100, score[3]*100), end='\r')
                            best_score = score[5]
                            best_score_std = score[6]
                            bscore = score
                            df_best = df
                            best_columns = df.columns
                            best_preprocess = score[4]
                            best_dict[alg_name] = [best_score, best_score_std]
                            method = alg(survey = args.survey, bands = bands, output_name = output_name, temp_path=temp_path, dataframe=df_best, path_to_csv=None, validation_set=False, output_path=output_path, sample_weight=weights, cv=cv, preprocessing=best_preprocess, n_jobs=args.nodes)
                            _, y_pred, y_test = method.model()
                            y_pred_dict[alg] = y_pred
                            y_test_dict[alg] = y_test
                        
                        break
                    
                    best_dict.to_cs(path + 'output/%s/%s/files/'%(args.survey, output_name) + 'Best_scores_' + output_name + '.csv', index=False)
                    # score = method.score()
                    
                    print('---------------------------------------------------------------')
                    print("%s: "%alg_name + "Sigma: {:.3f} ± {:.4f}, outlier rate: {:.3f} ± {:.3f} % ".format(bscore[0], bscore[1], bscore[2]*100, bscore[3]*100))
                    if best_score < global_score:
                        global_score = best_score
                        global_score_std = best_score_std
                        gscore = bscore
                        best_alg = alg_name
                        df_global = df_best
                        global_columns = best_columns
                        global_preprocess = best_preprocess
                        print('[NEW BEST] %s'%best_alg + ' score: {:.3f} ± {:.3f}'.format(global_score, global_score_std))
                    print('---------------------------------------------------------------')

                best_dict.sort_values(by = 'score', axis = 1, inplace=True)
                print(best_dict.head())

                df_best = df_global
                alg = algs[best_alg]
                method = alg(survey = args.survey, bands = bands, output_name = output_name, temp_path=temp_path, dataframe=df_best, path_to_csv=None, validation_set=False, output_path=output_path, sample_weight=weights, cv=cv, preprocessing=args.preprocess, n_jobs=args.nodes)
                if feature_importance == True:  
                    if best_alg != 'ANN':
                        method.permutation()
                if plot == True:
                    method.plot(lim=1.8)
                
                print('---------------------------------------------------------------')
                print('[BEST] preprocess: %s'%global_preprocess)
                print('[BEST] score: {:.3f} ± {:.3f}'.format(global_score, global_score_std))
                print(list(global_columns))
                print("[%s] "%args.algorithm + "%s: "%best_alg + "Sigma: {:.3f} ± {:.4f}, outlier rate: {:.3f} ± {:.3f} % ".format(gscore[0], gscore[1], gscore[2]*100, bscore[3]*100))
                print('---------------------------------------------------------------')

                


            else:
            
                try:
                    alg = algs[args.algorithm]
                except:
                    raise TypeError('MLM is not defined')

                if weights == True:
                    cat = MakeCatalogs(args.survey, bands, temp_path, output_name, output_path)
                    weights = cat.compute_weights(df, column = 'r')
                elif type(weights) == str:
                    weights = np.load(weights)
                else:
                    weights = None

                best_score = 1

                print('[Feature engineering]')
                print('---------------------------------------------------------------')

                for df in df_list:
                    method = alg(survey = args.survey, bands = bands, output_name = output_name, temp_path=temp_path, dataframe=df, path_to_csv=None, validation_set=False, output_path=output_path, sample_weight=weights, cv=cv, preprocessing=args.preprocess, n_jobs=args.nodes)
                    # method.plot(lim=1.8)
                    # method.permutation()

                    # df = method.filter()

                    # df.drop(columns=['r-z'], inplace=True)
                    score = method.score(df)

                    print(list(df.columns))
                    print('[preprocess] %s'%score[4])
                    print('[%s '%args.algorithm + 'score] {:.3f} ± {:.3f}'.format(score[5], score[6]))
                    if score[5] < best_score:
                        print('[NEW BEST]')
                        print("%s: "%args.algorithm + "Sigma: {:.3f} ± {:.4f}, outlier rate: {:.3f} ± {:.3f} % ".format(score[0], score[1], score[2]*100, score[3]*100))
                        best_score = score[5]
                        best_score_std = score[6]
                        bscore = score
                        df_best = df
                        best_columns = df.columns
                        best_preprocess = score[4]
                
                    # break


                method = alg(survey = args.survey, bands = bands, output_name = output_name, temp_path=temp_path, dataframe=df_best, path_to_csv=None, validation_set=False, output_path=output_path, sample_weight=weights, cv=cv, preprocessing=args.preprocess, n_jobs=args.nodes)
                if feature_importance == True:
                    if args.algorithm != 'ANN':
                        method.permutation()
                if plot == True:
                    method.plot(lim=1.5)
                if params.morph_importance == True and params.weights == False and args.algorithm == 'RF':
                    method.morph_importance(df_best)

                print('---------------------------------------------------------------')
                print('[BEST] preprocess: %s'%best_preprocess)
                print('[BEST] score: {:.3f} ± {:.3f}'.format(best_score, best_score_std))
                print(list(best_columns))
                print("%s: "%args.algorithm + "Sigma: {:.3f} ± {:.4f}, outlier rate: {:.3f} ± {:.3f} % ".format(bscore[0], bscore[1], bscore[2]*100, bscore[3]*100))
                print('---------------------------------------------------------------')


            #------------------------------------------------------------------#
            # # # # # OPTIMIZE LEARNING ALGORITHMS # # # # #
            #------------------------------------------------------------------#

            if args.optimize == 'HyperOpt' or args.optimize == 'RandomSearch' or args.optimize == 'GridSearch':

                # GenFiles = GenerateFiles(args.survey, bands, path, output_name, output_path=output_path)
                # GenFiles.make_directories(output=True)

                # path_to_csv = params.path_to_csv
                # max_evals = params.max_evals
                weights = params.weights
                # cv = params.cv

                algs = {'RF': RandomForestOptimizer, 'SVR': SVROptimizer, 'XGB': XGBoostOptimizer, 'KRR': KRROptimizer, 'ANN': ANNOptimizer}
                try:
                    alg = algs[args.algorithm]
                except:
                    raise ValueError('Method does not have an optimization algorithm')

                if weights == True:
                    cat = MakeCatalogs(args.survey, bands, temp_path, output_name, output_path)
                    weights = cat.compute_weights(df_best, column = 'r')
                elif type(weights) == str:
                    weights = np.load(weights)
                else:
                    weights = None

                print('[%s] optimization'%args.optimize)

                # if args.algorithm == 'ANN':

                    # ML = LearningAlgorithms(survey = args.survey, bands = bands, path_to_csv = path_to_csv, output_name = output_name, validation_set=True)
                    # X_train, X_val, X_test, Y_train, Y_val, Y_test = ML.data()
                    # X_train, Y_train, X_val, Y_val = data()

                    # trials = Trials()
                    # _, best_model = optim.minimize(model=model,data=data,algo=tpe.suggest, max_evals=max_evals, trials=trials)

                    # Y_pred = best_model.predict(X_test, verbose = 0)
                    # print(type(Y_pred), type(Y_test))
                    # sigma, eta = sigma_eta(Y_test.to_numpy().flatten(), Y_pred.flatten())
                    
                    # print("%s Opt : "%args.algorithm + "Sigma: {:.3f}, outlier rate: {:.3f} % ".format(sigma, eta*100))      

                    # ML.plot_zphot_zspec(Y_pred.flatten(), method='ANN_Opt', lim=1.8)

                    # ML = LearningAlgorithms(survey = args.survey, bands = bands, path_to_csv = path_to_csv, output_name = output_name, output_path=output_path, sample_weight=weights, cv=cv, preprocessing=args.preprocess, n_jobs=args.nodes)
                    # df = ML.dataframe()
                    # ML.plot_corrmat(df)
                    # ModelOptimizer = alg(survey = args.survey, bands = bands, output_name = output_name, dataframe=df, path_to_csv=None, validation_set=False)
                    # _, sigma, eta = ModelOptimizer.best_params(max_evals=10)
                    # print("%s Opt : "%args.algorithm + "Sigma: {:.3f}, outlier rate: {:.3f} % ".format(sigma, eta*100))    

                # if path_to_csv is None:
                # path_to_csv = output_path + 'output/' + args.survey  + '/' + output_name + '/files/' + output_name + '.csv'
                # ML = LearningAlgorithms(survey = args.survey, bands = bands, dataframe=df_best, output_name = output_name, output_path=output_path, sample_weight=weights, cv=cv, preprocessing=args.preprocess, n_jobs=args.nodes)
                # df, df_unmatched = ML.merge_cfis_r_cfht_u_medium_deep_i_g_z()

                # ML.plot_corrmat(df_best, figure_name=args.algorithm+'_best_corrmat')
                ModelOptimizer = alg(survey = args.survey, bands = bands, output_name = output_name, dataframe=df_best, path_to_csv=None, validation_set=False, output_path=output_path, sample_weight=weights, cv=cv, preprocessing=best_preprocess, n_jobs=args.nodes)
                
                # ModelOptimizer.debug()

                _, sigma, eta, score = ModelOptimizer.best_params(max_evals=max_evals, method=args.optimize)
                print('---------------------------------------------------------------')
                print('[BEST OPT] score: {:.3f}'.format(score))
                print("%s %s : "%(args.algorithm, args.optimize) + "Sigma: {:.3f}, outlier rate: {:.3f} % ".format(sigma, eta*100))
                print('---------------------------------------------------------------')

                # elif path_to_csv is not None:
                #     ML = LearningAlgorithms(survey = args.survey, bands = bands, path_to_csv = path_to_csv, output_name = output_name, output_path=output_path, sample_weight=weights, cv=cv, preprocessing=args.preprocess, n_jobs=args.nodes)
                #     df = ML.dataframe()
                #     ML.plot_corrmat(df)  
                #     ModelOptimizer = alg(survey = args.survey, bands = bands, output_name = output_name, dataframe=df, path_to_csv=None, validation_set=False, output_path=output_path, sample_weight=weights, cv=cv, preprocessing=args.preprocess, n_jobs=args.nodes)
                #     _, sigma, eta = ModelOptimizer.best_params(max_evals=max_evals, method=args.optimize)
                #     print("%s %s : "%(args.algorithm, args.optimize) + "Sigma: {:.3f}, outlier rate: {:.3f} % ".format(sigma, eta*100))

                # else:          
                #     ML = LearningAlgorithms(survey = args.survey, bands = bands, path_to_csv = path_to_csv, output_name = output_name)
                #     df = ML.dataframe()
                #     df = ML.preprocess(df, method = args.preprocess)
                #     ML.plot_corrmat(df)
                #     ModelOptimizer = alg(survey = args.survey, bands = bands, output_name = output_name, dataframe=df, path_to_csv=False, validation_set=False)
                #     _, sigma, eta = ModelOptimizer.best_params(max_evals=max_evals, method=args.optimize)
                #     print("%s %s : "%(args.algorithm, args.optimize) + "Sigma: {:.3f}, outlier rate: {:.3f} % ".format(sigma, eta*100))
        
            
#------------------------------------------------------------------#
# # # # # UNIONS # # # # #
#------------------------------------------------------------------#

    elif args.survey == 'unions_deprecated':

        spectral_path = '/home/mkilbing/astro/data/CFIS/spectro_surveys/'
        spectral_names = ['data_DR14_LRG_N', 'data_DR14_LRG_S', 'galaxy_DR12v5_CMASSLOWZTOT_North', 'galaxy_DR12v5_CMASSLOWZTOT_South','sdss_main_gal']
        # spectral_names = ['sdss_main_gal']

        spectral_surveys = ['SDSS', 'SDSS', 'eBOSS', 'eBOSS', 'SDSS_2']
        # spectral_surveys = ['SDSS_2']

        output_name = 'CFIS_matched_eBOSS_SDSS_catalog_RUIZ'
        # output_name = 'CFIS_matched_SDSS_2_catalog_RUIZ'
        output_path = path

        temp_path = '/n17data/jimenez/temp/'

        bands = ['R', 'U', 'I', 'Z']

        # out_dir = os.listdir("/n17data/jimenez/shaperun_unions/output_%s/"%(spectral_surveys[i]))[-1]
        # path_to_tile_run = '/n17data/jimenez/shaperun/'
        # input_path = path_to_tile_run + args.survey + '/%s/output/%s/paste_cat_runner/output/'%(spectral_surveys[i], out_dir)
        # paste_dir = os.listdir(input_path)
        


        if args.clean == True:
            GenFiles = GenerateFiles(args.survey, bands, temp_path)
            GenFiles.clean_temp_directories()
            GenFiles.make_directories()

        elif args.make == True:
            cat = MakeCatalogs(args.survey, bands, temp_path)
            # vignet = [False, False, False, False, False]
            for i in range(len(spectral_names)):
                cat.make_survey_catalog(spectral_path, spectral_names[i])
                out_dir = os.listdir("/n17data/jimenez/shaperun_unions/output_%s/"%(spectral_surveys[i]))[-1]
                paste_dir = os.listdir('/n17data/jimenez/shaperun_unions/output_%s/%s/paste_cat_runner/output/'%(spectral_surveys[i], out_dir))
                input_path = '/n17data/jimenez/shaperun_unions/output_%s/%s/paste_cat_runner/output/'%(spectral_surveys[i], out_dir)
                Parallel(n_jobs=args.nodes)(delayed(cat.make_catalog)(p, paste_dir, input_path, spectral_names[i], vignet=False) for p in tqdm(range(len(paste_dir))))

        elif args.join == True:
            cat = MakeCatalogs(args.survey, bands, temp_path)
            cat.merge_catalogs(output_name, vignet=False)

        elif args.generate_plots == True:
            GenPlot = GeneratePlots(args.survey, bands, temp_path, csv_name=output_name, spectral_names=spectral_names)
            # GenPlot.plot_d2d()
            GenPlot.plot_matched_r_i_i_z()
            GenPlot.plot_matched_u_r_r_i()
            GenPlot.plot_matched_z_spec_hist()
            # GenPlot.plot_unmatched_r_i_i_z()
            # GenPlot.plot_unmatched_u_r_r_i()
            GenPlot.plot_unmatched_z_spec_hist()

    # if args.survey != 'unions' or args.survey != 'ps3pi_cfis':
    #     print("Survey must either be 'unions' or 'ps3pi_cfis'")
        # raise SyntaxError("Survey must either be 'unions' or 'ps3pi_cfis'")



