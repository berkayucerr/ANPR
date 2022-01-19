from traceback import print_exception
import fonks as fonk

try:
    choice=input("Image_:")
    if(choice!=""):
        img=fonk.GetImage(choice)
        plate=fonk.filters(img)
        print(plate)
except Exception as e:
    print(e)