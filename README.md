## Inspiration
In multiple situations, people tend to unintentionally generate comments that are hateful or offensive. How can they ensure that their comments remain respectful and inclusive?

This question inspired me to create ToxiGuard, an AI-powered tool that classifies text as hateful, offensive, or neutral. With ToxiGuard, users can identify potentially harmful language and make informed adjustments to their messages and promote a healthy online environment.

## What it does
The user can enter any text they like, and the app will use BERT to tokenize the text and feed the tokenized text into the BERT model that was trained through the hate speech/offensive language csv dataset to classify whether the text is hateful, offensive, or neither.

## How we built it
First, I uploaded and processed a Hate Speech/Offensive Language dataset from Kaggle. After preprocessing the dataset, I used Hugging Face's BERT Base model for tokenization. This allows the model to handle unseen and rare words effectively by breaking them down into subword units, ensuring that the text is in a format that the model can process. Once tokenized, I fine-tuned the model using TFBERT from Hugging Face for the classification task. The dataset was split into training and validation sets, which enabled me to monitor the model's performance during training. To optimize the model, I employed the Adam optimizer with a learning rate scheduler, which improved convergence and ensured effective fine-tuning.

## Challenges we ran into
One challenge I faced was that the model was taking 30 minutes to train. To address this, I utilized the A100 GPU on Google Colab, which significantly improved training speed. With the A100 GPU, the training time was reduced to just 5 minutes per session.

## Accomplishments that we're proud of
The model was able to achieve a 91% validation accuracy, along with a validation recall of 93%. This means the model was able to correctly 91% of 2000 unseen datasets along with identifying the target class correctly 93% of the time. 

## What we learned
I learned how to perform classification tasks using Hugging Face models like BERT in combination with TensorFlow. This experience helped me understand the process of tokenizing text, fine-tuning models, and evaluating performance metrics to improve classification accuracy.

## What's next for ToxiGuard
In addition to classifying whether text is offensive, hateful, or neither, I aim to develop a model that can rewrite hateful or offensive text into more positive language. This would allow users to save time by automatically generating more constructive versions of their messages.
