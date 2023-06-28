import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

finger_tips =[8, 12, 16, 20]
thumb_tip= 4

while True:
    ret,img = cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape
    results = hands.process(img)
    hand_landmarks=results.multi_hand_landmarks

    if hand_landmarks:
        landmarks=hand_landmarks[0].landmark
        finger_fold_status=[]
        for landmarks in hand_landmarks:
            lm_list=[]
            for id ,lm in enumerate(landmarks.landmark):
                lm_list.append(lm)
            for i in finger_tips:
                x,y = int(lm_list[i].x*w),int(lm_list[i].y*h)
                cv2.circle(img,(x,y),15,(255,0,0),cv2.FILLED)
                if lm_list[i].x<lm_list[i-3].x:
                    cv2.circle(img,(x,y),15,(0,255,0),cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)
            if all(finger_fold_status):
                if lm_list[thumb_tip].y < lm_list[thumb_tip-1].y < lm_list[thumb_tip-2].y:
                    cv2.putText(img,"Like",(20,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)
                if lm_list[thumb_tip].y > lm_list[thumb_tip-1].y > lm_list[thumb_tip-2].y:
                    cv2.putText(img,"Dislike",(20,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)

            mp_draw.draw_landmarks(img, landmarks,mp_hands.HAND_CONNECTIONS)
            mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0,0,255),2,2),
            mp_draw.DrawingSpec((0,255,0),4,2)

    cv2.imshow("hand tracking", img)

    key = cv2.waitKey(1)
    if key == 32:
        break
     
cv2.destroyAllWindows()
