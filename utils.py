from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
import numpy as np

def evaluate_and_log_results(true_labels, pred_labels):
    if len(true_labels) == 0 or len(pred_labels) == 0:
        return

    try:
        print(f"accuracy: {accuracy_score(true_labels, pred_labels)}")
        print(f"f1_score macro: {f1_score(true_labels, pred_labels, average='macro')}")
        print(f"recall macro: {recall_score(true_labels, pred_labels, average='macro')}")
        print(f"precision macro: {precision_score(true_labels, pred_labels, average='macro')}")
        print(f"f1_score binary: {f1_score(true_labels, pred_labels, average='binary')}")
        print(f"recall binary: {recall_score(true_labels, pred_labels, average='binary')}")
        print(f"precision binary: {precision_score(true_labels, pred_labels, average='binary')}")
        print(f"f1_score micro: {f1_score(true_labels, pred_labels, average='micro')}")
        print(f"recall micro: {recall_score(true_labels, pred_labels, average='micro')}")
        print(f"precision micro: {precision_score(true_labels, pred_labels, average='micro')}")

        cnf_matrix = confusion_matrix(true_labels, pred_labels)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        print(f"TP: {TP}")
        print(f"FP: {FP}")
        print(f"cnf_matrix: {cnf_matrix}")
        tn, fp, fn, tp = cnf_matrix.ravel()
        print(f"ravel: {tn, fp, fn, tp}")

        # positive abnormal
        ################## 如果标签反了，交换 tn 和 tp，fp 和 fn
        tn, fp, fn, tp = tp, fn, fp, tn
        print(f"ravel: {tn, fp, fn, tp}")
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        print(f"fpr: {fpr}")
        print(f"fnr: {fnr}")

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)
        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)

        print(f"all TPR: {TPR}")
        print(f"all TNR: {TNR}")
        print(f"all PPV: {PPV}")
        print(f"all NPV: {NPV}")
        print(f"all FPR: {FPR}")
        print(f"all FNR: {FNR}")
        print(f"all FDR: {FDR}")
        print(f"AUC: {roc_auc_score(true_labels, pred_labels)}")

        # 设定感兴趣的FPR区间最大值 (max_fpr)
        max_fpr = 0.05 # DarkNet
        # max_fpr = 0.5

        # 计算在max_fpr下的原始部分AUC (pAUC)
        pAUC = roc_auc_score(true_labels, pred_labels, max_fpr=max_fpr)

        print(f"SPAUC: {pAUC}")
    except Exception as e:
        print(e)
        return