import re
def get_words(file_path):
      with open(file_path, mode='r', encoding='utf8') as f:
          lines = f.readlines()
      return [re.split('\s+', line) for line in lines]


def evalute(src, pred, ref, test_name =''):
    total_ref_j_list = []
    total_pred_j_list = []
    total_src_j_list = []

    if isinstance(src, str):
        src = [src]
        pred = [pred]
        ref = [ref]

    for src_line, pred_line, ref_line in zip(src, pred, ref):
        for s, p, r in zip(src_line, pred_line, ref_line):
            total_pred_j_list.append(p.strip())
            total_ref_j_list.append(r.strip())
            total_src_j_list.append(s.strip())

    def safe_division(n, d):
        return n / d if d > 0 else 0

    pred_edits_ps = total_pred_j_list
    gold_edits_ps = total_ref_j_list
    x_ps = total_src_j_list

    print(pred_edits_ps[0], gold_edits_ps[0], x_ps[0])


    d_tp, d_tn, d_fp, d_fn = 0, 0, 0, 0  # detection
    c_tp, c_tn, c_fp, c_fn = 0, 0, 0, 0  # correction

    for i, (x, gold, pred) in enumerate(zip(x_ps, gold_edits_ps, pred_edits_ps)):
        for o_word, g_word, e_word in zip([x], [gold], [pred]):
            o_word = o_word.strip()
            g_word = g_word.strip()
            e_word = e_word.strip()
            if o_word == g_word:  # no error
                if o_word == e_word:
                    d_tn += 1  # no error, not detected an error
                else:
                    d_fp += 1  # no error, but detected an error

                if g_word == e_word:
                    c_tn += 1  # no error, not corrected
                else:
                    c_fp += 1  # no error, but corrected
            else:  # error; o_word != g_word
                if o_word == e_word:
                    d_fn += 1  # error, but not detected an error
                else:
                    d_tp += 1  # error, detected an error

                if g_word == e_word:
                    c_tp += 1  # error, corrected accurately
                else:
                    c_fn += 1  # error, not/inaccurately corrected           

    d_recall = safe_division(d_tp, d_tp + d_fn)
    d_prec = safe_division(d_tp, d_fp + d_tp)
    c_recall = safe_division(c_tp, c_tp + c_fn)
    c_prec = safe_division(c_tp, c_fp + c_tp)

    detection_accuracy = safe_division((d_tn + d_tp), (d_tn + d_tp + d_fn + d_fp)) * 100
    detection_F1 = safe_division((2 * d_recall * d_prec * 100), (d_recall + d_prec))
    detection_F0_5 = safe_division(((1 + 0.5 ** 2) * d_recall * d_prec * 100), (d_recall + (0.5 ** 2) * d_prec))
    corrections_accuracy = safe_division((c_tn + c_tp), (c_tn + c_tp + c_fn + c_fp)) * 100
    corrections_F1 = safe_division((2 * c_recall * c_prec * 100), (c_recall + c_prec))
    corrections_F0_5 = safe_division(((1 + 0.5 ** 2) * c_recall * c_prec * 100), (c_recall + (0.5 ** 2) * c_prec))

    return {
        'test': test_name,
        'Detection TP': d_tp,
        'Detection TN': d_tn,
        'Detection FP': d_fp,
        'Detection FN': d_fn,
        'Detection Accuracy': detection_accuracy,
        'Detection Recall': d_recall * 100,
        'Detection Precision': d_prec * 100,
        'Detection F1': detection_F1,
        'Detection F0.5': detection_F0_5,
        'Correction TP': c_tp,
        'Correction TN': c_tn,
        'Correction FP': c_fp,
        'Correction FN': c_fn,
        'Correction Accuracy': corrections_accuracy,
        'Correction Recall': c_recall * 100,
        'Correction Precision': c_prec * 100,
        'Correction F1': corrections_F1,
        'Correction F0.5': corrections_F0_5,
    }

def colors(token, color='green'):
    c_green = '\033[92m'  # green
    c_red = '\033[91m'  # red
    c_close = '\033[0m'  # close
    if color=='green': 
        return c_green + token + c_close
    elif  color=='red':
        return c_red + token + c_close
print(colors('red', color='red'))
