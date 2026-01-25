import os
# Force TensorFlow to use the legacy Keras behavior for TF Hub compatibility
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # REQUIRED
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
#from imblearn.over_sampling import RandomOverSampler

df=pd.read_csv("hf://datasets/AmaanP314/youtube-comment-sentiment/youtube-comments-sentiment.csv")
print('load sccuess')
print(df.head())

print(df.isnull().sum())
print(df.shape)

df=df.dropna()

le=LabelEncoder()
df['Sentiment']=le.fit_transform(df['Sentiment'])

print(df['Sentiment'].value_counts())

df['combine_text']=df['VideoTitle']+" [SEP]comment: "+df['CommentText']

x=df['combine_text'].values
y=df['Sentiment'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

batch_size=8
train_ds=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_ds = train_ds.shuffle(len(x_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

preprocess_url="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"


def build_classifier_model():

    text_input=tf.keras.Input(shape=(),dtype=tf.string)
    preprocess_layer=hub.KerasLayer(preprocess_url)
    encoder_inputs=preprocess_layer(text_input)

    encoder_layer=hub.KerasLayer(encoder_url,trainable=True)
    outputs=encoder_layer(encoder_inputs)
    
    net=outputs['pooled_output']
    net=tf.keras.layers.Dropout(0.1)(net)
    net=tf.keras.layers.Dense(3,activation='softmax',dtype='float32')(net)
    
    return tf.keras.Model(text_input,net)
    
model=build_classifier_model()

model.compile(

    loss='sparse_categorical_crossentropy', 
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    metrics=['accuracy']

)

epoch_number=3
model.fit(train_ds,validation_data=test_ds,epochs=epoch_number)

tess_loss,test_accuracy=model.evaluate(test_ds)

print("test accuracy:",test_accuracy)

y_pred=model.predict(test_ds)
y_pred=np.argmax(y_pred,axis=1)

model.save('youtube_sentiment.h5')
print('model saved successfully')

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred,target_names=le.classes_))










