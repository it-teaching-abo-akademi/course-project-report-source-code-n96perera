#include <Arduino.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "model.h" // Your model header file
#include "input_image.h" // Your input image header file

// Define the size of the tensor arena
const int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Define the resolver for all TensorFlow Lite operations
static tflite::AllOpsResolver resolver;

// Initialize the TensorFlow Lite model pointer
const tflite::Model* tflite_model = nullptr;

// Define the interpreter
static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize);

// Get input and output tensors
TfLiteTensor* input = static_interpreter.input(0);
TfLiteTensor* output = static_interpreter.output(0);

void setup() {
  tflite::InitializeTarget();
  Serial.begin(9600);
  delay(20); // Give time to open serial monitor
  
  // Initialize the TensorFlow Lite model
  tflite_model = tflite::GetModel(model);

 
  // Allocate tensors
  if (static_interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Error allocating tensors!");
    while (1);
  }

  // Debug prints for tensor shapes
  Serial.print("Input tensor type: ");
  Serial.println(input->type);
  Serial.print("Input tensor dimensions: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  Serial.print("Output tensor type: ");
  Serial.println(output->type);
  Serial.print("Output tensor dimensions: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();
}

void loop() {
  // Load input image into input tensor
  for (int i = 0; i < input->bytes; i++) {
    input->data.uint8[i] = input_image[i];
  }

  // Run inference
  if (static_interpreter.Invoke() != kTfLiteOk) {
    Serial.println("Error running inference!");
    while (1);
  }

  // Find the class with the highest probability
  float max_prob = 0.0;
  int max_idx = -1;
  for (int i = 0; i < output->bytes / sizeof(float); i++) {
    float prob = output->data.f[i];
    if (prob > max_prob) {
      max_prob = prob;
      max_idx = i;
    }
  }

  // Print the results
  Serial.print("Predicted class: ");
  Serial.print(max_idx);
  Serial.print(", Probability: ");
  Serial.println(max_prob);

  // Delay before next inference
  delay(1000);
}