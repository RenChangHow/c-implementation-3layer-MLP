/**
 * NUS EE4218 AY2018/19 Project
 * An C implemetation of a 3-layer Perceptron Network for Multi-class Classification.
 * Notes - The code is specially designed for Wine Dataset, with 13 input features and 3 classes.
 *       - This code is an end-to-end classification model. It will be trained using training set, and make predictions on test set.
 *       - Prediction accuracy on test set will be displayed. A csv file containing predicted results will be generated.
 *       - Dataset paramters (e.g. input_shape, output_shape, #training data, #test data, etc)
 *         are hard-coded for simplicity, which can be modified for other classification tasks and datasets.
 *       - Model hyper-parameters (e.g. alpha(learning rate), #hidden layer neurons, max epochs, etc) can be modified for improvements.
 *       - Mini-Batches are generated using default pseudo-random generator function rand(), which can be improved.
 *       - Basic SGD method with no momentum or decay is used as optimizer, which can improved.
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/**
 * Network Hyper-parameters
 */
# define NUM_INPUT_NEURONS 13
# define NUM_HIDDEN_NEURONS 10
# define NUM_OUTPUT_NEURONS 3
# define alpha 0.001 // Learning rate
# define BATCH_SIZE 20
# define NUM_EPOCH 1000
# define ERROR_TOLERRANCE 0.1

/**
 * Training / Test dataset parameters
 */
# define TRAINING_DATA_SIZE 160
# define TEST_DATA_SIZE 18
# define INPUT_DIMENSION NUM_INPUT_NEURONS
# define OUTPUT_DIMENSION NUM_OUTPUT_NEURONS
# define TRAINING_DATASET_PATH "train.txt"
# define TEST_DATASET_PATH "test.txt"


/** 
 * Training / Test dataset
 */
int training_dataset[TRAINING_DATA_SIZE][INPUT_DIMENSION];
int training_label[TRAINING_DATA_SIZE];
int test_dataset[TEST_DATA_SIZE][INPUT_DIMENSION];
int test_label[TEST_DATA_SIZE];
int predicted_label[TEST_DATA_SIZE];

/** 
 * Weight matrices
 * - IH: from input layer to hidden layer
 * - HO: from hidden layer to output layer
 */
float weights_IH[NUM_HIDDEN_NEURONS][NUM_INPUT_NEURONS+1];
float weights_HO[NUM_OUTPUT_NEURONS][NUM_HIDDEN_NEURONS+1];

/**
 * Network neurons
 */
float input_neurons[NUM_INPUT_NEURONS+1];
float hidden_neurons[NUM_HIDDEN_NEURONS+1];
float output_neurons[NUM_OUTPUT_NEURONS];
//float training_batch_input[BATCH_SIZE][INPUT_DIMENSION+1];
//float training_batch_hidden[BATCH_SIZE][NUM_HIDDEN_NEURONS+1];
//float training_output_batch[BATCH_SIZE][OUTPUT_DIMENSION];
//int training_batch_label[BATCH_SIZE];

/**
 * Derivative of loss w.r.t a certain neuron (stored for back propagation)
 */
float delta_hidden_neurons[NUM_HIDDEN_NEURONS+1];
float delta_output_neurons[NUM_OUTPUT_NEURONS];

/**
 * Helper Functions
 */
void read_training_dataset(void);
void read_test_dataset(void);
void write_predicted_test_label(void);
float get_rand(void);
void init_weights(void);
void train_network(void);
void network_predict(void);
float feed_forward_train(int index);
int feed_forward_test(int index);
void back_propagate();
float activate_neuron(float x);
float delta_activate_neuron(float x);
int random_sampling(void);
//void generate_mini_batch(void);

// debug helper functions
void print_training_dataset(void) {
    int i, j;
    for (i = 0; i < TRAINING_DATA_SIZE; i++) {
        printf("training data #%d - %d | ", i, training_label[i]);
        for (j = 0; j < INPUT_DIMENSION; j++) {
            printf("%d ", training_dataset[i][j]);    
        }
        printf("\n");
    }
    return;
}

void print_weights_IH(void) {
    int i, j;
    printf("Weights_IO: \n");
    for (i = 0; i < NUM_HIDDEN_NEURONS; i++) {
        for (j = 0; j < NUM_INPUT_NEURONS + 1; j++) {
            printf("%f ", weights_IH[i][j]);
        }
        printf("\n");
    }
}

void print_weights_HO(void) {
    int i, j;
    printf("Weights_HO: \n");
    for (i = 0; i < NUM_OUTPUT_NEURONS; i++) {
        for (j = 0; j < NUM_HIDDEN_NEURONS + 1; j++) {
            printf("%f ", weights_HO[i][j]);
        }
        printf("\n");
    }
}

void debug(void) {
    //print_training_dataset();
    print_weights_IH();
    print_weights_HO();
}


int main() {
    train_network();
    network_predict();
    //debug();
    return 0;
}



/**
 * Function to read in the training / test dataset csv file, including data + labels
 */
void read_training_dataset(void) {
    int col_idx, row_idx;
    int curr;
    printf("Please send training data. Wating for training data...\n");
    // FILE *file = fopen("train_data.txt", "r");
    // for (row_idx = 0; row_idx < TRAINING_DATA_SIZE; row_idx++) {
    //     fscanf(file, "%d,", &curr);
    //     printf("#%d label = %d ", row_idx, curr);
    //     training_label[row_idx] = curr;
    //     for (col_idx = 0; col_idx < INPUT_DIMENSION; col_idx++) {
    //         fscanf(file, "%d,", &curr);
    //         training_dataset[row_idx][col_idx] = curr;
    //     }
    // }
    // fclose(file);
    // printf("Training dataset read successfully.\n");
    // //print_training_dataset();
    // return ;
    
    int i = 0, idx;
    int data[INPUT_DIMENSION+1];
    FILE *train=fopen(TRAINING_DATASET_PATH,"r");
    while( i < TRAINING_DATA_SIZE) {
        fscanf(train,"%d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",&data[0],&data[1],&data[2],&data[3],&data[4],&data[5],&data[6],&data[7],&data[8],&data[9],&data[10],&data[11],&data[12],&data[13]);
        training_label[i] = data[0];
		for (idx = 1; idx < INPUT_DIMENSION + 1; idx++) {
            //printf("%d ", data[idx]);
            training_dataset[i][idx-1] = data[idx];
        }
		i++;
    }
    fclose(train);
    return;
}


void read_test_dataset(void) {
    int col_idx, row_idx;
    printf("Please send test data. Wating for test data...\n");
    // FILE *file = fopen("test_data.csv", "r");
    // for (row_idx = 0; row_idx < TEST_DATA_SIZE; row_idx++) {
    //     fscanf(file, "%d", &test_label[row_idx]);
    //     for (col_idx = 0; col_idx < INPUT_DIMENSION; col_idx++) {
    //         fscanf(file, "%d", &test_dataset[row_idx][col_idx]);    
    //     }
    // }
    // fclose(file);

    int i = 0, idx;
    int data[INPUT_DIMENSION+1];
    FILE *test=fopen(TEST_DATASET_PATH,"r");
    while( i < TEST_DATA_SIZE ) {
        fscanf(test,"%d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",&data[0],&data[1],&data[2],&data[3],&data[4],&data[5],&data[6],&data[7],&data[8],&data[9],&data[10],&data[11],&data[12],&data[13]);
        test_label[i] = data[0];
		for (idx = 1; idx < INPUT_DIMENSION + 1; idx++) {
            test_dataset[i][idx-1] = data[idx];
        }
		i++;
    }
    fclose(test);
    printf("Testing tataset read successfully.\n");
    return ;
}


/**
 * Write the predicted labels of test dataset to an csv file.
 */
void write_predicted_test_label(void) {
    int idx;
    printf("Writing predicted labels to result.csv ...\n");
    FILE *file = fopen("result.csv", "w");
    for (idx = 0; idx < TEST_DATA_SIZE; idx++) {
        fprintf(file, "%d\n", predicted_label[idx]);
    }
    fclose(file);
    printf("Writing predicted labels complete!\n");
    return;
}


/**
 * Generate random number between 0~1
 */
float get_rand(void) {
    //srand(time(NULL));
    return ((float)rand()) / (float)RAND_MAX;
}


/**
 * Initialize the weights of network
 */
void init_weights(void) {
    int i, j;
    for (i = 0; i < NUM_HIDDEN_NEURONS; i++) {
        for (j = 0; j < NUM_INPUT_NEURONS + 1; j++) {
            weights_IH[i][j] = get_rand() - 0.5;
        }
    }

    for (i = 0; i < NUM_OUTPUT_NEURONS; i++) {
        for (j = 0; j < NUM_HIDDEN_NEURONS + 1; j++) {
            weights_HO[i][j] = get_rand() - 0.5;
        }
    }
    return;
}


/**
 * Initialize neurons values of network
 */
void init_neurons(void) {
    int i;
    for (i = 0; i < NUM_HIDDEN_NEURONS + 1; i++) {
        hidden_neurons[i] = 0;
    }
    for (i = 0; i < NUM_OUTPUT_NEURONS; i++) {
        output_neurons[i] = 0;
    }
}


// /**
//  * Generate a mini batch of training data, of size (BATCH_SIZE)
//  */
// void generate_mini_batch(void) {
//     int i, j, idx;
//     for (i = 0; i < BATCH_SIZE; i++) {
//         idx = rand() % BATCH_SIZE;
//         training_batch_input[i][0] = 1;
//         for (j = 1; j < INPUT_DIMENSION + 1; j++) {
//             training_batch_input[i][j] = training_dataset[idx][j];
//         }
//         training_batch_label[i] = training_label[i];
//     } 
// }


/**
 * Randomly sample a training data as the input neurons
 * @return index of sampled data in training dataset
 */
int random_sampling(void) {
    int i, idx;
    //srand(time(NULL));
    idx = rand() % TRAINING_DATA_SIZE;
    input_neurons[0] = 1;
    for (i = 1; i < NUM_INPUT_NEURONS + 1; i++) {
        input_neurons[i] = training_dataset[idx][i];
    }
    return idx;
}


/**
 * The activation function of neurons.
 * @param x: value of input from former layer.
 * @return output of neuron to next neuron
 */
float activate_neuron(float x) {
    // Sigmoid Function
    return 1.0 / (1 + exp(-x));

    // ReLU activation function
    //return (x > 0) ? x : 0;
}

float delta_activate_neuron(float x) {
    // Derivative of Sigmoid Function
    return x * (1-x);

    // Derivative of ReLU 
    //return (x > 0) ? 1 : 0;
}


/**
 * Feed forward a sample data into the network for training, and calculate the loss.
 * @param data_index: the index of current sample in training dataset.
 * @return the loss based on current model and sample data.
 */
float feed_forward_train(int data_index) {
    int i, idx;
    float sum = 0;
    int gt_label = training_label[data_index] - 1;
    // Input layer to hidden layer
    hidden_neurons[0] = 1;
    for (idx = 1; idx < NUM_HIDDEN_NEURONS + 1; idx++) {
        hidden_neurons[idx] = 0;
        for (i = 0; i < NUM_INPUT_NEURONS + 1; i++) {
            hidden_neurons[idx] += input_neurons[i] * weights_IH[idx-1][i];
        }
        hidden_neurons[idx] = activate_neuron(hidden_neurons[idx]);
    }

    // Hidden layer to output layer
    for (idx = 0; idx < NUM_OUTPUT_NEURONS; idx++) {
        output_neurons[idx] = 0;
        for (i = 0; i < NUM_HIDDEN_NEURONS + 1; i++) {
            output_neurons[idx] += hidden_neurons[i] * weights_HO[idx][i];
        }
    }

    // Update derivative of cross-entropy loss w.r.t each output neuron
    for (idx = 0; idx < NUM_OUTPUT_NEURONS; idx++) {
        output_neurons[idx] = exp(output_neurons[idx]);
        sum += output_neurons[idx];
    }
    for (idx = 0; idx < NUM_OUTPUT_NEURONS; idx++) {
        if (idx == gt_label) {
            delta_output_neurons[idx] = (output_neurons[idx] / sum) - 1;
        } else {
            delta_output_neurons[idx] = output_neurons[idx] / sum;
        }
    }

    // Calculate Loss of current sample data (Softmax Loss)
    return -log(output_neurons[gt_label]/sum);
}

/**
 * Feed forward a sample data for testing, and returns the predicted classification.
 * @param data_index: index of current sample data in test dataset.
 * @return predicted label [1 or 2 or 3]
 */
int feed_forward_test(int data_index) {
    int i, idx, label;
    float max_score;

    // Input layer to hidden layer
    hidden_neurons[0] = 1;
    for (idx = 1; idx < NUM_HIDDEN_NEURONS + 1; idx++) {
        hidden_neurons[idx] = 0;
        for (i = 0; i < NUM_INPUT_NEURONS + 1; i++) {
            hidden_neurons[idx] += input_neurons[i] * weights_IH[idx-1][i];
        }
        hidden_neurons[idx] = activate_neuron(hidden_neurons[idx]);
    }

    // Hidden layer to output layer
    for (idx = 0; idx < NUM_OUTPUT_NEURONS; idx++) {
        output_neurons[idx] = 0;
        for (i = 0; i < NUM_HIDDEN_NEURONS + 1; i++) {
            output_neurons[idx] = output_neurons[idx] + hidden_neurons[i] * weights_HO[idx][i];
        }
    }

    // find the neuron with highest prediction score
    max_score = output_neurons[0];
    label = 1;
    printf("#%d - a1: %f | ", data_index, max_score);
    for (idx = 1; idx < NUM_OUTPUT_NEURONS; idx++) {
        printf("a%d: %f | ",idx+1, output_neurons[idx]);
        if (output_neurons[idx] > max_score) {
            max_score = output_neurons[idx];
            label = idx + 1;
        }
    }
    printf("predicted: %d truth: %d ", label, test_label[data_index]);
    if (label == test_label[data_index]) {
        printf("\n");
    } else {
        printf("x\n");
    }

    // log the prediction of current sample
    predicted_label[data_index] = label;

    return (label == test_label[data_index]) ? 1 : 0;
}


/**
 * Backpropagate the network to update weights according to current loss.
 * @param data_index: the index of current sample in generated mini-batch of training sample.
 */
void back_propagate(void) {
    int i, j;

    // back_propagate from output layer to hidden layer
    delta_hidden_neurons[0] = 0;
    for (j = 1; j < NUM_HIDDEN_NEURONS + 1; j++) { // for each neuron in hidden layer except the first fake one
        delta_hidden_neurons[j] = 0;
        for (i = 0; i < NUM_OUTPUT_NEURONS; i++) {
            delta_hidden_neurons[j] += delta_output_neurons[i] * weights_HO[i][j];
        }
    }

    // update hidden-output weights_HO 
    for (i = 0; i < NUM_OUTPUT_NEURONS; i++) {
        for (j = 0; j < NUM_HIDDEN_NEURONS + 1; j++) {
            weights_HO[i][j] = weights_HO[i][j] - alpha * (delta_output_neurons[i] * hidden_neurons[j]);
        }
    }

    // back propagate through the activation function of hidden layer
    for (i = 1; i < NUM_HIDDEN_NEURONS + 1; i++) {
        delta_hidden_neurons[i] = delta_activate_neuron(hidden_neurons[i]) * delta_hidden_neurons[i];
    }

    // update input-hidden weights_IH
    for (i = 1; i < NUM_HIDDEN_NEURONS + 1; i++) {
        for (j = 0; j < NUM_INPUT_NEURONS + 1; j++) {
            weights_IH[i-1][j] = weights_IH[i-1][j] - alpha * (delta_hidden_neurons[i] * input_neurons[j]);
        }
    }
    return;
}


/**
 * Training the network.
 */
void train_network(void) {
    float loss = 0;
    int batch, counter = 0, epoch = 0;
    int index;
    read_training_dataset();
    init_weights();
    printf("Starting to train network...\n");
    do {
        loss = 0;
        for (counter = 1; counter <= TRAINING_DATA_SIZE / BATCH_SIZE; counter++) {
            for (batch = 0; batch < BATCH_SIZE; batch++) {
                init_neurons();
                index = random_sampling();
                //printf("Epoch: %d Batch: %d Sample #%d - index = %d\n",epoch, counter, batch, index);
                loss += feed_forward_train(index);
                back_propagate();
                //printf("Epoch: %d Batch: %d - ",epoch, counter);
                //print_weights_IH();
            }
        }
        epoch++;
        loss = loss / TRAINING_DATA_SIZE;
        printf("Epoch No.: %d - Loss: %f \n", epoch, loss);
    } while (epoch < NUM_EPOCH && loss > ERROR_TOLERRANCE); // (loss > ERROR_TOLERRANCE) is to prevent overfitting. 
    printf("Network training completed!\n");
    return;
}


/**
 * Use the trained network to make predictions.
 */
void network_predict(void) {
    int i, idx;
    int sum = 0;
    read_test_dataset();
    printf("Starting to make predictions...\n");
    for (idx = 0; idx < TEST_DATA_SIZE; idx++) {
        input_neurons[0] = 1;
        for (i = 1; i < NUM_INPUT_NEURONS + 1; i++) {
            input_neurons[i] = test_dataset[idx][i];
        }
        sum += feed_forward_test(idx);
    }
    printf("Network prediction complete!\n");
    printf("Prediction Accuracy: %f\n", (sum*1.0/TEST_DATA_SIZE));
    write_predicted_test_label();
}