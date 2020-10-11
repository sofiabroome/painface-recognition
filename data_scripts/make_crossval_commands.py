import sys
sys.path.append('..')
import os
import helpers


def get_train_test(dataset):

    horses_lps = ['aslan', 'brava', 'herrera', 'inkasso', 'julia',
                       'kastanjett', 'naughty_but_nice', 'sir_holger']
    horses_pf = ['horse_1', 'horse_2', 'horse_3', 'horse_4', 'horse_5', 'horse_6']
    
    if dataset == 'all':
        train_horses = horses_lps + horses_pf
        test_horses = horses_lps + horses_pf
    if dataset == 'pf':
        train_horses = horses_pf
        test_horses = horses_pf
    if dataset == 'lps':
        train_horses = horses_lps
        test_horses = horses_lps

    return train_horses, test_horses

def get_val(dataset, test_subject):

    if dataset == 'all':
        val_horses = ['kastanjett', 'horse_5']
        if test_subject == 'horse_5':
            val_horses = ['kastanjett', 'horse_1']
        if test_subject == 'kastanjett':
            val_horses = ['brava', 'horse_5']
    if dataset == 'pf':
        val_horses = ['horse_5']
        if test_subject == 'horse_5':
            val_horses = ['horse_1']
    if dataset == 'lps':
        val_horses = ['kastanjett']
        if test_subject == 'kastanjett':
            val_horses = ['brava']

    return val_horses

def make_commands(dataset, nb_repetitions):
    
    train_horses, test_horses = get_train_test(dataset)

    for rep in range(nb_repetitions):
        commands = []
        for test_subject in test_horses:

            val_horses = get_val(dataset, test_subject)
            train_subjects = [x for x in train_horses
                                if x is not test_subject
                                and x not in val_horses]

            str_com = ['sbatch --export ']
            str_com+= ['CONFIG_FILE=', config_file]
            str_com+= [',TRAIN_SUBJECTS=', '/'.join(train_subjects)]
            str_com+= [',VAL_SUBJECTS=', '/'.join(val_horses)]
            str_com+= [',TEST_SUBJECTS=', test_subject]
            str_com+= ' eqpain.sbatch'
            str_com = ''.join(str_com)
            commands.append(str_com)
            print(str_com)

    out_file = os.path.join('../run_scripts', job_name + '.sh')
    helpers.write_file(out_file, commands)


def make_jobarray_configs(dataset, nb_repetitions):
    
    train_horses, test_horses = get_train_test(dataset)
    output_dir = os.path.join('../run_scripts', job_name)
    helpers.mkdir(output_dir)

    counter_config = 1
    for rep in range(nb_repetitions):
        for ind, test_subject in enumerate(test_horses):
            commands = []

            val_horses = get_val(dataset, test_subject)
            train_subjects = [x for x in train_horses
                                if x is not test_subject
                                and x not in val_horses]
            commands.append('bash')
            commands.append('CONFIG_FILE=' + config_file)
            commands.append('TRAIN_SUBJECTS=' + '/'.join(train_subjects))
            commands.append('VAL_SUBJECTS=' + '/'.join(val_horses))
            commands.append('TEST_SUBJECTS=' + test_subject)
            print(commands, '\n')
            # config_filename = 'config-' + ('%02d' % (counter_config)) + '.sh'
            config_filename = 'config-' + str(counter_config) + '.sh'
            out_file = os.path.join(output_dir, config_filename)
            print(out_file)
            print('\n')
            print('\n')
            
            helpers.write_file(out_file, commands)
            counter_config += 1


def main():
    # make_commands(dataset=dataset_str,
    #               nb_repetitions=nb_reps)
    make_jobarray_configs(dataset=dataset_str,
                          nb_repetitions=nb_reps)

if __name__=='__main__':

    dataset_str = 'pf'
    # dataset_str = 'lps'
    # dataset_str = 'all'

    nb_reps = 3

    model = '2stream'
    # model = 'clstm1'

    if model == '2stream':
        config_file = 'configs/config_2stream_{}.py'.format(dataset_str)
    if model == 'clstm1':
        config_file = 'configs/config_clstm.py'
    job_name = 'configs_to_run_{}_{}_crossval'.format(model, dataset_str)
    main()

