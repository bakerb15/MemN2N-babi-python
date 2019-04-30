file_orig = 'original_training_log_task1.txt'
file_pt = 'pytorch_training_log_task1.txt'

orig_lines = []
pt_lines = []


def parse(ulin):
    sp0 = ulin.split('|')
    epoch = sp0[0].split()[1]
    train_err = sp0[1].split()[1]
    valerr = sp0[2].split()[1]
    return epoch, train_err, valerr

with open(file_orig, 'r') as f:
    for line in f.readlines():
        if 'train error:' in line:
            orig_lines.append(line)

with open(file_pt, 'r') as f:
    for line in f.readlines():
        if 'train error:' in line:
            pt_lines.append(line)


with open('train_log_both.csv', 'w') as f:
    f.write('epoc, oirig_train, orig_val, pytorch_train, pytorch_val\n')
    for i in range(len(orig_lines)):
        orig_ep, orig_te, orig_ve = parse(orig_lines[i])
        pt_ep, pt_te, pt_ve = parse(pt_lines[i])
        if orig_ep == pt_ep:
            f.write('{}, {}, {}, {}, {}\n'.format(orig_ep, orig_te, orig_ve, pt_te, pt_ve))
