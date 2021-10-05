
import numpy as np

from timeit import default_timer as timer

from tqdm import tqdm

import util

def ssc_CV(C_reg, clf_type, dataset, train_trial_num):   

    board = util.create_board()
        
    data = dataset['data']
    y_stim = data['y_stim'][0][0]

    ### Reorganize the matrix
    _, X_sorted, y_sorted = util.process_data(data)
    flash_mat = util.process_stim(y_stim) #which row/column being flashed

    ### Some constants for data splitting
    flash_per_seq = 12
    seq_per_trial = 10
    train_trial_num = 20
    seq_num, trial_num, test_trial_num = util.constant_for_split(X_sorted.shape[0], flash_per_seq, seq_per_trial, train_trial_num)


    new_X = np.reshape(X_sorted, (seq_num,flash_per_seq,120))
    new_y = np.reshape(y_sorted, (seq_num,flash_per_seq))

    char_true, _, _ = util.char_list(flash_mat, new_y) # the character flashed for each sequence
    char_true = np.reshape(char_true, (trial_num,seq_per_trial))
    
    char_label = char_true[:,0] # Correct character for each trial
    og_label = np.reshape(char_label, (7,5)) # just for reference, not used in our code
    
    # For 35 trials, each trial has 10 sequences, each sequence has 12 flashes
    indices = np.arange(seq_num)
    test_index = indices[train_trial_num*10:]
    total_train_index = indices[:train_trial_num*10]
    
    mean_accuracy = np.zeros(C_reg.shape[0])
    for j, cc in enumerate(tqdm(C_reg, desc ="Progress")):

        ### Calculate accuracy: 
        char_correct = []
        cv_accuracy = np.zeros(train_trial_num)  
            
        for i in range(train_trial_num): # loop thru the trial and leave one out
            
            valid_index = indices[10*i:10*i+10]
            train_index = np.array([x for x in indices[:train_trial_num*10] if x not in valid_index]) # training set, all but the validation trial
            
            ### Initialize the testing/validation set
            X_train = new_X[train_index]
            y_train = new_y[train_index]
            X_valid = new_X[valid_index]
            y_valid = new_y[valid_index]
            
            flash_valid = flash_mat[valid_index]
            
            ### reshape data for clf
            new_X_train = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1],120))
            new_y_train = np.reshape(y_train, (y_train.shape[0]*y_train.shape[1]))
            new_X_valid = np.reshape(X_valid, (X_valid.shape[0]*X_valid.shape[1],120))
            new_y_valid = np.reshape(y_valid, (y_valid.shape[0]*y_valid.shape[1]))
                        
            ### Train the model
            clf = util.apply_classifier(clf_type, cc)
    
            clf.fit(new_X_train, new_y_train)
            
            ### Validate w/ every single sequence - Calculate the probability of each char 
            
            # if (clf_type == 'SWLDA'):
            #     y_score = clf.test(new_X_valid, labels = new_y_valid) 
            # else:
            # y_score = clf.predict_proba(new_X_valid)[:, 1] # the probability of the target row/col being flashed
            y_score = clf.decision_function(new_X_valid) # the probability of the target row/col being flashed

            y_score_mat = np.reshape(y_score, (np.int64(y_score.shape[0]/12),12)) #reshape into #sequences x 12 matrix
            
            mean_char_proba = np.zeros((6,6))
            
            util.update_char_proba(mean_char_proba, y_score_mat, flash_valid, seq_per_trial, flash_per_seq)
            
            board = util.create_board()
            char_hat = board[np.where(mean_char_proba == np.max(mean_char_proba))]
            
            if (char_hat == char_label[i]): 
                char_correct.append(char_hat) #Identify how many chars get correct
            
            ### Calculate classifier accuracy:
            # if (clf_type == 'SWLDA'):
            #     new_y_valid_pred = np.zeros(y_score.shape)
            #     # = np.array([0] if i < 0 else [1] for i in y_score)
            #     loc_accuracy = np.count_nonzero(new_y_valid_pred == new_y_valid)/new_y_valid.shape[0]
            # else:    
            loc_accuracy = clf.score(new_X_valid, new_y_valid)
        
            cv_accuracy[i] = loc_accuracy # pseudo-accuracy
            
        char_correct = np.array(char_correct).flatten()
        char_accuracy = len(char_correct)/train_trial_num
        
        print(f'The regularization param C is :{cc}')
        print(f'The accuracy of character selection: {char_accuracy}') # n/20
        print(f'The cv accuracy of the classifier is: {np.mean(cv_accuracy)}')
    
        mean_accuracy[j] = np.mean(char_accuracy)
        
    C_index = np.argmax(mean_accuracy)
    C_best = C_reg[C_index]
    C_best = 0.01
    print(f'The best reguarlization parameter C is {C_best}')
    
    ### Testing data
    X_train_total = new_X[total_train_index]
    y_train_total = new_y[total_train_index]
    X_test = new_X[test_index]
    y_test = new_y[test_index]
    
    flash_test = flash_mat[test_index]
    
    # reshape data for clf
    new_X_train_total = np.reshape(X_train_total, (X_train_total.shape[0]*X_train_total.shape[1],120))
    new_y_train_total = np.reshape(y_train_total, (y_train_total.shape[0]*y_train_total.shape[1]))
    new_X_test = np.reshape(X_test, (X_test.shape[0]*X_test.shape[1],120))
    new_y_test = np.reshape(y_test, (y_test.shape[0]*y_test.shape[1]))
    
    start = timer()
    
    clf = util.apply_classifier(clf_type, cc = C_best)
    
    clf.fit(new_X_train_total, new_y_train_total)
    
    ### Validate w/ every single sequence - Calculate the probability of each char 
    # if (clf_type == 'SWLDA'):
    #     y_score_total = clf.test(new_X_test, labels = new_y_test) 
    # else: 
    # y_score_total = clf.predict_proba(new_X_test)[:, 1] # the probability of the target row/col being flashed
    y_score_total = clf.decision_function(new_X_test) # the probability of the target row/col being flashed

        
    y_score_mat_total = np.reshape(y_score_total, (np.int64(y_score_total.shape[0]/flash_per_seq),flash_per_seq)) #reshape into #sequences x 12 matrix
    
    y_score_mat_total = np.reshape(y_score_mat_total, (np.int64(y_score_mat_total.shape[0]/seq_per_trial),seq_per_trial, flash_per_seq))
    flash_test = np.reshape(flash_test, (np.int64(flash_test.shape[0]/seq_per_trial),seq_per_trial, flash_per_seq))
    
    ### Loop through all test characters to check accuracy
    char_correct_test = []
    for i in range(test_trial_num):
        
        y_score_trial = y_score_mat_total[i]
        flash_test_trial = flash_test[i]
        
        mean_char_proba = np.zeros((6,6))
        
        util.update_char_proba(mean_char_proba, y_score_trial, flash_test_trial, seq_per_trial, flash_per_seq)
        
        char_hat = board[np.where(mean_char_proba == np.max(mean_char_proba))]
        
        if (char_hat == char_label[i+train_trial_num]): 
            char_correct_test.append(char_hat) #Identify how many chars get correct

    char_correct_test = np.array(char_correct_test).flatten()
        
    train_accuracy = clf.score(new_X_train_total, new_y_train_total)        
    test_accuracy = clf.score(new_X_test, new_y_test)
    # if (clf_type == 'SWLDA'):
    #     accuracy = 
    
    print(f'The accuracy of test character selection: {len(char_correct_test)/test_trial_num}') # n/10
    print(f'The total training accuracy of the classifier is: {train_accuracy}')
    print(f'The testing accuracy of the classifier is: {test_accuracy}')
    print(f'Runtime is: {timer() - start}')
    
    ### Plot ROCs: 
    # if (clf_type == 'SWLDA'):
    #     y_score_train = clf.test(new_X_train_total, labels = new_y_train_total) ### te_score: probability score
    # else: 
    # y_score_train = clf.predict_proba(new_X_train_total)[:, 1]
    y_score_train = clf.decision_function(new_X_train_total)
    
    util.plotROC(new_y_train_total, y_score_train, new_y_test, y_score_total, clf_type)