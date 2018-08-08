 function EVAL = Evaluate(TP, TN, FP, FN)
% Input: TP, TN, FP, FN
% Output: EVAL = Row matrix with all the performance measures

p = TP + FN;
n = FP + TN;
N = p + n;

tp = TP
tn = TN
fp = n-tn;
fn = p-tp;

tp_rate = tp/p;
tn_rate = tn/n;

accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
f_measure = 2*((precision*sensitivity)/(precision + sensitivity));
Youden = sensitivity + specificity - 1;
DOR = (tp/fp)/(fn/tn);

EVAL = [accuracy sensitivity specificity precision f_measure Youden DOR];