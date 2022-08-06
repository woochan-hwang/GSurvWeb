# LOAD DEPENDENCY ----------------------------------------------------------
import streamlit as st

from itertools import product
from main.models.SVM import SupportVectorClassifier, SupportVectorRegression
from main.models.RF import RandomForest
from main.models.CART import RegressionTree
from main.models.MLP import MultiLayerPerceptron

# MAIN SCRIPT --------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def process_options(model, selected_features, selected_label, censoring, rfe):
    model.get_input_features(selected_features)
    model.get_label_feature(label_feature=selected_label, censoring=censoring)
    model.rfe = rfe
    return model

def experiment(file, verbose):

    st.header('Experiment setting')
    selected_label = st.selectbox('Prediction target:', ['Survival time [days]', 'Failure within 1 year [y/n]'])
    censor_state = st.checkbox('Left censoring to 1 year', value=True)

    if selected_label == 'Failure within 1 year [y/n]':
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader('SVM parameters')
                svm_model = SupportVectorClassifier()
                svm_model.class_weight = 'balanced'
                # iterable parameters
                c_param = st.select_slider('C parameter:', options=svm_model.c_param_list, value=(0.01, 1),
                                           help='The C parameter in a support vector machine is a regularisation parameter. The strength of the regularization is inversely proportional to C.')
                kernel = st.multiselect('Kernel:', ['linear', 'poly', 'rbf', 'sigmoid'], default=['linear'],
                                        help='Recursive feature elimination function only works for linear kernels.')
                svm_iterables = svm_model.get_iterable_model_options(kernel, c_param=c_param)


            with col2:
                st.subheader('RF parameters')
                rf_model = RandomForest()
                rf_model.class_weight = 'balanced_subsample'
                # iterable parameters
                n_estimators = st.select_slider('Number of trees:', options=rf_model.n_estimators_list, value=(10, 50))
                max_depth = st.select_slider('Depth of trees:', options=rf_model.max_depth_list, value=(3, 5))
                criterion = st.multiselect('Criterion?', ['gini', 'entropy'], default='gini')
                rf_iterables = rf_model.get_iterable_model_options(criterion, n_estimators=n_estimators, max_depth=max_depth)

    elif selected_label == 'Survival time [days]':
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader('SVM parameters')
                svm_model = SupportVectorRegression()
                # iterable parameters
                c_param = st.select_slider('C parameter:', options=svm_model.c_param_list,
                                           value=(0.01, 1),
                                           help='The C parameter in a support vector machine is a regularisation parameter. The strength of the regularization is inversely proportional to C.')
                kernel = st.multiselect('Kernel:', ['linear', 'poly', 'rbf', 'sigmoid'], default=['linear'])
                svm_iterables = svm_model.get_iterable_model_options(kernel, c_param=c_param)

            with col2:
                st.subheader('CART parameters')
                rf_model = RegressionTree()
                # iterable parameters
                max_depth = st.select_slider('Depth of tree:', options=rf_model.max_depth_list, value=(3, 5))
                max_leaf_nodes = st.select_slider('Number of leaves:', options=rf_model.max_leaf_nodes_list, value=(3, 5))
                criterion = st.multiselect('Criterion?', ['mse', 'mae', 'friedman_mse', 'poisson'], default='mse')
                rf_iterables = rf_model.get_iterable_model_options(criterion, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)

        mlp_model = MultiLayerPerceptron()
        mlp_model.hidden_layer_sizes = (100,100)

    st.subheader('Feature selection')

    svm_model.get_data(file)
    rf_model.get_data(file)
    mlp_model.get_data(file)

    available_input_options = svm_model.get_input_options()  # return value is equal given same df for all subclass of BaseModel

    selected_features = st.multiselect('Input features (select at least one):', available_input_options)
    feature_elimination = st.checkbox('Recursive feature elimination?', value=True, help='Must select at least 2 features')

    svm_model = process_options(svm_model, selected_features, selected_label, censoring=censor_state, rfe=feature_elimination)
    rf_model = process_options(rf_model, selected_features, selected_label, censoring=censor_state, rfe=feature_elimination)
    mlp_model = process_options(mlp_model, selected_features, selected_label, censoring=censor_state, rfe=feature_elimination)

    if len(svm_model.input_features) == 0:
        st.warning('❗Please select at least one input feature to run analysis.')
        st.stop()

    st.write("### Dataset size")
    svm_model.process_input_options(verbose=False)
    rf_model.process_input_options(verbose=False)
    mlp_model.process_input_options(verbose=False)

    st.write("### Train / Test split")
    test_proportion = st.slider('Test set proprotion?', min_value=0.1, max_value=0.5, step=0.1, value=0.3)
    svm_model.train_test_split(test_proportion=test_proportion)
    rf_model.train_test_split(test_proportion=test_proportion, verbose=False)
    mlp_model.train_test_split(test_proportion=test_proportion, verbose=False)

    verbose = st.checkbox('Print training performance')

    if st.session_state['continue_state'] == False:
        st.session_state['continue_state'] = st.button('Continue')
        st.stop()

    if st.session_state['train_state'] == False:

        with st.spinner('training in progress...'):

            for params in list(product(*svm_iterables)):
                svm_model.kernel = params[0]
                svm_model.c_param = params[1]

                svm_model.build_estimator()
                svm_model.train(verbose=verbose)

                svm_model.evaluate(verbose=verbose)
                svm_model.save_log()

            if selected_label == 'Failure within 1 year [y/n]':

                for params in list(product(*rf_iterables)):
                    rf_model.criterion = params[0]
                    rf_model.max_depth = params[1]
                    rf_model.n_estimators = params[2]

                    rf_model.build_estimator()
                    rf_model.train(verbose=verbose)

                    rf_model.evaluate(verbose=verbose)
                    rf_model.save_log()

            elif selected_label == 'Survival time [days]':

                for params in list(product(*rf_iterables)):
                    rf_model.criterion = params[0]
                    rf_model.max_depth = params[1]
                    rf_model.max_leaf_nodes = params[2]

                    rf_model.build_estimator()
                    rf_model.train(verbose=verbose)

                    rf_model.evaluate(verbose=verbose)
                    rf_model.save_log()

                mlp_model.build_estimator()
                mlp_model.train(verbose=verbose)
                mlp_model.evaluate(verbose=verbose)
                mlp_model.save_log()

        st.success('Done!')
        st.session_state['train_state'] = True

    st.session_state['save_state'] = st.button('Save')

    if st.session_state['save_state']:
        svm_model.export_log()
        rf_model.export_log()
        mlp_model.export_log()
