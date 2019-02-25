import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
plt.ioff()



parser = argparse.ArgumentParser(description='inputs to train')
parser.add_argument('--folder_exp', type=str, default='', metavar='N',
                    help='path to outputs')
parser.add_argument('--validate', type=str, default='', metavar='N',
                    help='train or test')
parser.add_argument('--n_epochs', type=int, default=49, metavar='N',
                    help='number of epochs to eval')


main_path = '/media/SSD3/Astronomy/models/densenetOwn/5classes_problem/'
experiment = args.folder_exp
n_classes = 6
which_set = args.validate

for epoch in range(args.n_epochs):
    if which_set =='test':
        outputs = np.load(main_path + experiment + '/validation_' + str(epoch) + '_outputs.npy')
        labels = np.load(main_path + experiment + '/validation_'+ str(epoch) + '_targets.npy')
    elif which_set =='train':
        outputs = np.load(main_path + experiment + '/train_' + str(epoch)+'_outputs.npy')
        labels= np.load(main_path + experiment + '/train_' + str(epoch)+ '_targets.npy')

    best_precision = []
    best_recall = []
    best_F = []
    best_AP = []
    #labels = np.load(main_path + experiment + '/labels_best_model.npy')
    clases = ['BZ','AGN','CV','OTHER','SN','NON']
    clas_num = range(6)
    if n_classes ==6:
        for ii in range(n_classes):
            precision, recall, thresholds = precision_recall_curve(labels, outputs[:,ii], pos_label=ii)
            #AP = average_precision_score(labels, outputs[:,ii]) 
            F = 2*(precision*recall)/(precision + recall + 1e-06)
            pos_best_Fmeasure = np.argmax(F)
            F_best = np.max(F)
            best_F.append(F_best)
            best_precision.append(precision[pos_best_Fmeasure])
            best_recall.append(recall[pos_best_Fmeasure])
        metrics = np.stack((clas_num, best_F, best_precision, best_recall), axis=1)
        file_name = main_path + experiment + '/' + which_set + '_'+str(epoch)+'_metrics_per_class.txt'
        np.savetxt(file_name, metrics, fmt=['%u','%.4e', '%.4e', '%.4e'], header = 'C Fmeasu Precision Recall')
        F_avg = np.mean(np.array(best_F))
        F_std = np.std(np.array(best_F))
        BP_avg = np.mean(np.array(best_precision))
        BP_std =np.std(np.array(best_precision))
        BR_avg = np.mean(np.array(best_recall))
        BR_std = np.std(np.array(best_recall))
        name_global = main_path + experiment + '/' + which_set + '_avgMetrics.txt'
        avg_metrics = '{} F-measure: {:.4f} +/- {:.4f}, Precision: {:.4f} +/- {:.4f}, Recall: {:.4f} +/- {:.4f} \n'.format(epoch,F_avg, F_std,BP_avg,BP_std,BR_avg,BR_std)
        with open(name_global,'a') as f:
            f.write(avg_metrics)

    elif n_classes ==2:
        precision, recall, thresholds = precision_recall_curve(labels, outputs[:,1], pos_label=1)
        F = 2*(precision*recall)/(precision + recall + 1e-06)
        #AP = average_precision_score(labels, outputs[:,1]) 
        pos_best_Fmeasure = np.argmax(F)
        F_best = np.max(F)
        best_precision.append(precision[pos_best_Fmeasure])
        best_recall.append(recall[pos_best_Fmeasure])
        #best_AP.append(AP)
        best_F.append(F_best)

        avg_metrics = '{} F-measure: {:.4f}, Precision: {:.4f}, Recall: {:.4f} \n'.format(epoch,best_F, best_precision,best_recall)
        with open(file_name,'a') as f:
            f.write(avg_metrics)