from cytomine.cytomine import Cytomine
from cytomine.models.annotation import AlgoAnnotationTerm, Annotation
from shapely.wkt import loads
import numpy as np
import cv, cv2
import sys
from colour_deconvolution import deconvolve_im_array
from otsu_threshold_with_mask import otsu_threshold_with_mask
import copy
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import maximum_filter, minimum_filter
from time import localtime, strftime
import time
from cytomine_utilities.reader import Bounds, CytomineReader
from cytomine_utilities.wholeslide import WholeSlide
from cytomine_utilities.objectfinder import ObjectFinder
from cytomine_utilities.utils import Utils
import socket
import time

job_id = 62531467
image_id = 37167157

term_to_segment = [15054705,28792193]
cell_to_classify = 15054765



#working_path = './temp'
working_path = '/data/home/adeblire/tmp' 


MOD = np.array([[ 56.24850493,  71.98403122,  22.07749587], #purple cells
                [ 48.09104103,  62.02717516,  37.36866958], # noise (verifier utilite)
                [  9.17867488,  10.89206473,   5.99225756]]) #background (verifier utilite)
                
cell_max_area = 4000
cell_min_area = 800
cell_min_circularity = 0.85

        
border = 7

struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)) #change to circle

struct_elem = np.array( [[0, 0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0],], dtype = np.uint8)






conn = Cytomine('beta.cytomine.be', 'd8bee9aa-4d6e-4cb4-a7e8-2d3100f0ba24', 'a3227ede-6ec5-4711-b621-6c14b2f8da45', working_path, base_path = "/api/", verbose = False)


print "Create Job and UserJob..."
image_instance = conn.getImageInstance(image_id)

image_height = image_instance.height



software_id = 28886993
user_job = conn.addUserJob(software_id, image_instance.project)
#Switch to job Connection
conn.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
job = conn.getJob(user_job.job)
#job = conn.update_job_status(job, status_comment = "Create software parameters values...")
#job_parameters_values = conn.add_job_parameters(user_job.job, conn.getSoftware(id_software), pyxit_parameters)
job = conn.update_job_status(job, status = job.RUNNING, progress = 0, status_comment = "Loading data...")

start_time = time.time()

print "Connecting to Cytomine server"
print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
whole_slide = WholeSlide(conn.get_image_instance(image_id, True))
print "Done"

downloaded = False
while (not downloaded) :
    
    try : 
        
        annotations = conn.get_annotations(id_user = job_id, id_image = image_id, id_terms = term_to_segment, showWKT=True)
        downloaded = True
        
    except socket.error :
        
        print socket.error
        time.sleep(1)
        
    except socket.timeout :
        
        print socket.timeout
        time.sleep(1)


i = 0

geometries = []

nb_annotations = len(annotations.data())

crop_time = 0

colordec_time = 0

otsu_time = 0

filling_time = 0

prewatershed_time = 0

watershed_time = 0

filtering_time = 0

upload_time = 0

nb_cells = 0

nb_rejected = 0


for k, annotation in enumerate(annotations.data()):
   
    start = time.time()
    
    p = loads(annotation.location)
    
    minx, miny, maxx, maxy = int(p.bounds[0]), int(p.bounds[1]), int(p.bounds[2]), int(p.bounds[3])    
    annotation_window = Bounds(minx,image_height-maxy, maxx-minx, maxy-miny)
    
    
    url = annotation.get_annotation_alpha_crop_url(annotation.term[0], desired_zoom = 0)
        
    filename = "./temp/crop.png"
    
    
    downloaded = False
    while (not downloaded) :
        
        try : 
            
            conn.fetch_url_into_file(url, filename, is_image = False , override = True)
            downloaded = True
            
        except socket.error :
            
            print socket.error
            time.sleep(1)
            
        except socket.timeout :
            
            print socket.timeout
            time.sleep(1)
    
    end = time.time()
    
    crop_time = crop_time + end - start
    
       
    np_image = cv2.imread(filename, -1)
    
    if np_image is not None :
    
        alpha = np.array(np_image[:,:,3])
    
        image = np.array(np_image[:,:,0:3])
    
    else :

        print "reading error"

        continue
    
      
    temp =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    start = time.time()
                   
    im_dec = deconvolve_im_array(temp, MOD)
    
    end = time.time()
    
    colordec_time = colordec_time + end - start
    
    temp =  cv2.cvtColor(im_dec, cv2.COLOR_RGB2GRAY)
    
    start = time.time()
    
    otsu_threshold, internal_binary = otsu_threshold_with_mask(temp, alpha, cv2.THRESH_BINARY_INV)
            
    internal_binary_copy = copy.copy(internal_binary)
    
    contours2, hierarchy = cv2.findContours(internal_binary_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    end = time.time()
    
    otsu_time = otsu_time + end - start
    
    #filling inclusions without filling inter-cell space

    start = time.time()

    mask = np.zeros(internal_binary.shape)
    
    for i, contour in enumerate(contours2):
                
        if (hierarchy[0, i, 3] != -1): #internal contour
                                
            convex_hull = cv2.convexHull(contour)
        
            convex_area = cv2.contourArea(convex_hull)
        
            perimeter = cv2.arcLength(convex_hull,True)
            
            circularity = 4*np.pi*convex_area / (perimeter*perimeter)
        
            #visualisation
            #cv2.drawContours(np_image, contour, -1, (0,255,255), -1)
            
            if (convex_area < cell_max_area/10):
        
                #removing small objects

                cv2.drawContours(internal_binary, [convex_hull], -1, 255, -1)
            
            if (convex_area < cell_max_area/2) and (circularity > 0.8):
        
                #removing potential inclusions

                cv2.drawContours(internal_binary, [convex_hull], -1, 255, -1)
    
    end = time.time()
    
    filling_time = filling_time + end - start
    
    start = time.time()         
                   
    internal_binary = cv2.morphologyEx(internal_binary,cv2.MORPH_CLOSE,struct_elem, iterations = 1)
    
    internal_binary = cv2.morphologyEx(internal_binary,cv2.MORPH_OPEN,struct_elem, iterations = 2)
        
    dt = cv2.distanceTransform(internal_binary, cv2.cv.CV_DIST_L2, 3)
    
    dt = dt[0]
    
    im_dec[internal_binary == 0] = (255,255,255)
    
    #im_dec = cv2.dilate(im_dec,struct_elem) #filtre max ??
    
    #detection maxima locaux
    local_min_ind = maximum_filter(dt, (9,9) ) == dt
    
    #image markers
    markers = np.zeros(dt.shape).astype(np.uint8)
    
    #maxima locaux
    markers[local_min_ind] = 255
    
    #impose tous les markers sont a l'interieur du contour
    markers[internal_binary == 0] = 0
    
    markers = cv2.dilate(markers,struct_elem, iterations = 2)
    
    markers = markers.astype(np.int32)
    
    #labellise les composantes connexes 1...nbmarkers
    markers, nb_labels = label(markers, np.ones((3,3)))
    
    borders = cv2.dilate(internal_binary,struct_elem, iterations = 1)
    
    markers[borders == 0] = 0
    
    borders = borders - internal_binary
    
    #cadre blanc autour (pour eviter contour de la taille iamge?)
    markers[borders == 255] = nb_labels+2
    
    markers = cv2.copyMakeBorder(markers, border, border, border, border, cv2.BORDER_CONSTANT, value = nb_labels+2)
    
    im_dec = cv2.copyMakeBorder(im_dec, border, border, border, border, cv2.BORDER_CONSTANT, value = (255,255,255))
    
    end = time.time()
    
    prewatershed_time = prewatershed_time + end - start
    
    start = time.time()
    cv2.watershed(im_dec, markers)
    end = time.time()
    
    watershed_time = watershed_time + end - start
    
    #enleve cadre
    markers = markers[border:-border, border:-border]

    # repasse en opencv (pour compatibilit objectfinder)
    internal_binary = np.zeros(internal_binary.shape).astype(np.uint8)
    
    cv_image = cv.CreateImageHeader((internal_binary.shape[1], internal_binary.shape[0]), cv.IPL_DEPTH_8U, 1) 
    
    
      
    #calcul circularite, aire,... pour chaque contour pour filtrage
    for l in range(1,nb_labels + 1) :
    
        start = time.time() 
       
        mask = np.zeros(markers.shape).astype(np.uint8)
        
        mask[markers == l] = 255

        #mask = cv2.dilate(mask, struct_elem)
        
        #copie avant modification par findcontour pour garder image originale
        mask_copy = copy.copy(mask)
        
        cell_contour, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)              

        perimeter = cv2.arcLength(cell_contour[0],True)
        
        area = cv2.contourArea(cell_contour[0])
        
        circularity = 4*np.pi*area / (perimeter*perimeter)
        
        convex_hull = cv2.convexHull(cell_contour[0])
        
        convex_perimeter = cv2.arcLength(convex_hull,True)
        
        convex_area = cv2.contourArea(convex_hull)
        
        convexity = area / convex_area

        convex_circularity = 4*np.pi*convex_area / (convex_perimeter*convex_perimeter)
        
        end = time.time() 
        
        filtering_time = filtering_time + end - start
        
        if  (cell_min_area < area < cell_max_area) and (convex_circularity > cell_min_circularity) : 
            
            nb_cells = nb_cells + 1
            
            cv2.drawContours(mask, [convex_hull], -1, 255, -1)
            
            cv.SetData(cv_image, mask.tostring())
            
            cv_size = cv.GetSize(cv_image)
            
            components = ObjectFinder(cv_image).find_components()
            
            components = whole_slide.convert_to_real_coordinates(whole_slide, components, annotation_window, 0)
            
            geometries.extend(Utils().get_geometries(components))
            
            print "Uploading annotations..."
            
            print "Number of geometries: %d" % len(geometries)
            
            start = time.time()
            
            for geometry in geometries:
                uploaded = False
                while(not uploaded) :
                    try :
                        print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                        print "Uploading geometry %s" % geometry
                        annotation = conn.add_annotation(geometry, image_id)
                        uploaded = True
                    except socket.error :
                        print socket.error
                        time.sleep(1)
                    except socket.timeout :
                        print socket.timeout
                        time.sleep(1)


                    #print annotation
                if annotation:
                    annotated = False
                    while(not annotated):
                        try :
                            conn.add_annotation_term(annotation.id, cell_to_classify, cell_to_classify, 1.0, annotation_term_model = AlgoAnnotationTerm)
                            annotated = True
                        except socket.error :
                            print socket.error
                            time.sleep(1)
                            
                        except socket.timeout :
                            print socket.timeout
                            time.sleep(1)

            end = time.time()
            
            upload_time = upload_time + end - start



        else  : nb_rejected = nb_rejected + 1
            
        geometries = []
            

            
    print "number of annotations processed :", k,"/",nb_annotations
    print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())

end_time = time.time()
            
job = conn.update_job_status(job, status = job.TERMINATED, progress = 100, status_comment =  "test job complete")

print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())  

print "TOTAL time : ", end_time - start_time," s"

print "crop time : ",crop_time," s"

print "colordec time : ",colordec_time," s"

print "otsu time : ",otsu_time," s"

print "filling time : ",filling_time," s"

print "prewatershed time : ",prewatershed_time," s"

print "watershed time : ",watershed_time ," s"

print "filtering time : ",filtering_time ," s"

print "upload time : ",upload_time ," s"

print "number of annotations processed : ", k

print "number of cells : ", nb_cells

print "number of rejected zones : ", nb_rejected 

print "image : ", image_id

print "job : ", job.userJob


print "END"
 

