from email import message
import os
from cv2 import preCornerDetect
from numpy.lib.function_base import average
from tensorflow.python.keras.layers.pooling import AvgPool3D
from c3d import *
from classifier import *
from utils1.visualization_util import *
import smtplib, ssl
import datetime
import threading

def send_mail(message):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "teamdrishya.manthan@gmail.com"  # Enter your address
    receiver_email = "iamjeevank2000@gmail.com"  # Enter receiver address
    password = 'drishya123'  

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


def run_demo(only_graph):

    # port = 465  # For SSL
    # smtp_server = "smtp.gmail.com"
    # sender_email = "teamdrishya.manthan@gmail.com"  # Enter your address
    # receiver_email = "iamjeevank2000@gmail.com"  # Enter receiver address
    # password = 'drishya123'


    # context = ssl.create_default_context()
    # with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    #     server.login(sender_email, password)
    #     server.sendmail(sender_email, receiver_email, message)


    video_name = os.path.basename(cfg.sample_video_path).split('.')[0]

    # build models
    feature_extractor = c3d_feature_extractor()
    classifier_model = build_classifier_model()

    print("Models initialized")

    prediction_list = []

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        rgb_features = []
        frames = []
        cntr = 0
        
        while (cntr<params.features_per_bag):
            cntr+=1
            ret, frame = cap.read()
            if ret == True:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break
        num_frames = len(frames)
        print("FRAMES: ",num_frames)
        video_clips = sliding_window(frames, params.frame_count, params.frame_count)
        for i, clip in enumerate(video_clips):
            clip = np.array(clip)
            if len(clip) < params.frame_count:
                continue

            clip = preprocess_input(clip)
            rgb_feature = feature_extractor.predict(clip)[0]
            rgb_features.append(rgb_feature)

            print("Processed clip : ", i)
        rgb_features = np.array(rgb_features)

        # bag features
        rgb_feature_bag = interpolate(rgb_features, params.features_per_bag)

        # classify using the trained classifier model
        predictions = classifier_model.predict(rgb_feature_bag)


        if predictions.shape[0] == 1:
            predictions = predictions[:,0]
        else:
            predictions = np.array(predictions).squeeze()
        print("PREDICTION LENGTH: ", len(predictions))
        predictions = extrapolate(predictions, num_frames)

        # if predictions[0:len(predictions)] > 0.02:
        #     print(predictions[0:len(predictions)])
        
        save_path = os.path.join(cfg.output_folder, video_name + '.gif')
        # visualize predictions
        prediction_list.append(predictions)
        visualize_predictions(cfg.sample_video_path, predictions, save_path, only_graph, num_frames)

        if average(predictions[0:num_frames])*100000 > 40:
            op = 'ABNORMAL | '+str(average(predictions[0:num_frames])*100000)
            cv2.putText(frame, op, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2, cv2.LINE_AA)
            message = """\
            Subject: Abnormal Event Detected 

            CCTV-ID  : 2981\n
            Timestamp: {}.""".format(datetime.datetime.now()) 
            mail_thread = threading.Thread(target=send_mail, args=(message,))
            mail_thread.start()
        else:
            op = 'NORMAL | '+str(average(predictions[0:num_frames])*100000)
            cv2.putText(frame, op , (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 2, cv2.LINE_AA)

        cv2.imshow('Frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            x = range(0, len(prediction_list))
            y = prediction_list[::]
            plt.plot(x, y, '-')
            plt.savefig("output_camera.png")
            break


if __name__ == '__main__':
    run_demo(2)
