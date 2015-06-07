/*
 * Copyright (c) 2009-2015. Authors: Loic Rollus, University of Liege, http://www.cytomine.be/
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import com.vividsolutions.jts.io.*
import com.vividsolutions.jts.geom.*
import be.cytomine.client.models.*
import be.cytomine.client.*
import be.cytomine.client.collections.*;

int i = 0
def cytomineHost =  args[i++]
def cytominePublicKey = args[i++]
def cytominePrivateKey = args[i++]

def image = args[i++]
def user = args[i++]
def term = args[i++]
def minIntersectLength = Double.parseDouble(args[i++])
def bufferLength = Double.parseDouble(args[i++])

def minPointForSimplify = Long.parseLong(args[i++])
def minPoint = Long.parseLong(args[i++])
def maxPoint = Long.parseLong(args[i++])

def maxWith = Long.parseLong(args[i++])
def maxHeight = Long.parseLong(args[i++])

println "cytomineHost=$cytomineHost; cytominePublicKey=$cytominePublicKey; cytominePrivateKey=$cytominePrivateKey;"
println "image=$image; user=$user; term=$term; minIntersectLength=$minIntersectLength; bufferLength=$bufferLength;"

Cytomine cytomine = new Cytomine(cytomineHost, cytominePublicKey, cytominePrivateKey);


unionPicture(cytomine,image,user,term,bufferLength,minIntersectLength,minPointForSimplify,minPoint,maxPoint,maxWith,maxHeight)


public void unionPicture(def cytomine,def image, def user, def term,def bufferLength, def minIntersectLength,def minPointForSimplify,def minPoint,def maxPoint, def maxWith, def maxHeight) {
     //makeValidPolygon(image,user)
    println "minPoint=$minPoint maxPoint=$maxPoint"
     //http://localhost:8080/api/algoannotation/union?idImage=8120370&idUser=11974001&idTerm=9444456&minIntersectionLength=10&bufferLength=0&area=2000

    def areas = computeArea(cytomine,image,maxWith,maxHeight)

    def unionedAnnotation = []

     boolean restart = true

    //union by area
     int max = 100
     while(restart && (max>0)) {
         def result = unionArea(cytomine,image,user,term,bufferLength,minIntersectLength,areas,true)
         restart = result.restart
         unionedAnnotation.addAll(result.unionedAnnotation)
         max--
     }


    unionedAnnotation.unique()

    println "SIMPLIFY NOW"
    unionedAnnotation.each { idAnnotation ->
        try {
            Annotation based = getAnnotation(cytomine,idAnnotation)

            if(based && new WKTReader().read(based.get('location')).getNumPoints()>minPointForSimplify) {
                println "simplifyAnnotation=" + based.getId()

                cytomine.simplifyAnnotation(idAnnotation,minPoint,maxPoint)
            }

        }catch(Exception e) {
            println e
        }
    }
 }

private def computeArea(def cytomine, def idImage, def nbreAreaW, def nbreAreaH) {

    ImageInstance image = cytomine.getImageInstance(Long.parseLong(idImage))
    Double width = image.getLong('width')
    Double height = image.getLong('height')

    println "width=$width"
    println "height=$height"

    println "nbreAreaW=$nbreAreaW"
    println "nbreAreaH=$nbreAreaH"

//    Integer nbreAreaW =  Math.ceil(width/(double)maxW)
//    Integer nbreAreaH = Math.ceil(height/(double)maxH)

    def areaW = Math.ceil(width/(double)nbreAreaW)
    def areaH = Math.ceil(height/(double)nbreAreaH)

    println "nbreAreaW=$nbreAreaW"
    println "height=$height"

    def areas = []
    for(int i=0;i<nbreAreaW;i++) {
        for(int j=0;j<nbreAreaH;j++) {

            double bottomX = i*areaW
            double bottomY = j*areaH
            double topX = bottomX+areaW
            double topY = bottomY+areaH

            //println  bottomX + "x" + bottomY +" => " + topX + "x" + topY

            Coordinate[] boundingBoxCoordinates = [new Coordinate(bottomX, bottomY), new Coordinate(bottomX, topY), new Coordinate(topX, topY), new Coordinate(topX, bottomY), new Coordinate(bottomX, bottomY)]
            Geometry boundingbox = new GeometryFactory().createPolygon(new GeometryFactory().createLinearRing(boundingBoxCoordinates), null)
            areas <<  boundingbox
        }
    }
    areas
}


 private def unionArea(def cytomine,def image, def user, def term,def bufferLength, def minIntersectLength,def areas, boolean byArea) {
     println "unionArea..."
     List intersectAnnotation = null
//     if(byArea) {
         intersectAnnotation = intersectAnnotationArea(cytomine,image,user,term,bufferLength,minIntersectLength,areas)
//     } else {
//         intersectAnnotation = intersectAnnotationFull(cytomine,image,user,term,bufferLength,minIntersectLength,areas.size())
//     }

     //println "intersectAnnotation=$intersectAnnotation"
     println intersectAnnotation.size()

     boolean mustBeRestart = false
     def unionedAnnotation = []

     intersectAnnotation.eachWithIndex { it, indice ->
         HashMap<Long, Long> removedByUnion = new HashMap<Long, Long>(1024)

             long idBased = it[0]
             //check if annotation has be deleted (because merge), if true get the union annotation
             if (removedByUnion.containsKey(it[0]))
                 idBased = removedByUnion.get(it[0])

             long idCompared = it[1]
             //check if annotation has be deleted (because merge), if true get the union annotation
             if (removedByUnion.containsKey(it[1]))
                 idCompared = removedByUnion.get(it[1])

             Annotation based
             Annotation compared
             println "$idBased vs $idCompared"
             based = getAnnotation(cytomine,idBased)
             compared = getAnnotation(cytomine,idCompared)

            println "get ok..."
             if (based && compared && based.get('id') != compared.get('id')) {
                 mustBeRestart = true

                 try {
                     basedLocation = new WKTReader().read(based.get('location'))
                     comparedLocation = new WKTReader().read(compared.get('location'))

                     if(bufferLength) {
                         basedLocation = basedLocation.buffer(bufferLength)
                         comparedLocation = comparedLocation.buffer(bufferLength)
                     }

                     basedLocation = combineIntoOneGeometry([basedLocation,comparedLocation])
                     basedLocation = basedLocation.union()

                     if(bufferLength) {
                         basedLocation =  basedLocation.buffer(-bufferLength)
                     }

                     based.set('location',basedLocation.toText())

                     //based.location = simplifyGeometryService.simplifyPolygon(based.location.toText()).geometry
                     //based.location = based.location.union(compared.location)
                     removedByUnion.put(compared.get('id'), based.get('id'))

                     //save new annotation with union location
                     unionedAnnotation << based.get('id')
    //                 if(based.get('algoAnnotation')) {
                         //TODO: UPDATE  based
                     println "$indice/${intersectAnnotation.size()} ===> edit ${based.getId()} and delete ${compared.getId()}"
                     println "edit..."
                     cytomine.editAnnotation(based.getId(),based.get('location'))
                         //algoAnnotationService.saveDomain(based)
                         //remove old annotation with data
                         //TODO: DELETE COMPARED
                     println "delete..."
                     cytomine.deleteAnnotation(compared.getId())
                     } catch(Exception e) {
                         e.printStackTrace()
                    }

                  println "ok"

             }
     }
     return [restart: mustBeRestart, unionedAnnotation : unionedAnnotation]
 }


//TODO: create getAnnotationDomain

static Annotation getAnnotation(def cytomine, def id) {
     try {
         return cytomine.getAnnotation(id)
     } catch(Exception e) {
         return null
     }
}

static Geometry combineIntoOneGeometry( def geometryCollection ){
     GeometryFactory factory = new GeometryFactory();

     // note the following geometry collection may be invalid (say with overlapping polygons)
     GeometryCollection geometryCollection1 =
          (GeometryCollection) factory.buildGeometry( geometryCollection );

     return geometryCollection1.union();
 }


 private List intersectAnnotationArea(def cytomine, def image, def user, def term, def bufferLength, def minIntersectLength,def areas) {
     def data = []
     println "intersectAnnotation..."

     def filters = [:]
     filters.put("user",user);
     filters.put("image",image);
     if (term!=0) { filters.put("term",term.toString());}
     filters.put("showTerm","true");
     filters.put("showWKT","true");

     AnnotationCollection annotations = cytomine.getAnnotations(filters);

     println "annotations=${annotations.size()}"
     println "annotations=${annotations.toURL()}"

     def annotationsMap = [:]
     for(int i=0;i<annotations.size();i++) {
         def annotation = annotations.get(i)
         if(annotation.getId().toString().equals("27809914")) {
             println "##################"
         }
         annotationsMap.put(annotation.getId(),new WKTReader().read(annotation.get('location')).buffer(bufferLength))
     }

     println "annotationsMap=${annotationsMap.size()}"


     def annotationPerArea = [:]
     areas.eachWithIndex { area, i ->
         annotationPerArea.put(i,[])
     }

     def list = annotationsMap.entrySet().asList()

     println "List=${list.size()}"

     list.eachWithIndex { entry, i ->
         def form = entry.value
         if(entry.key.toString().equals("27809914")) {
             println "-----------------"
             println "form.isvalid=${form.isValid()}"
         }
         if(form.isValid()) {
             for(int j=0;j<areas.size();j++) {
                 if(form.intersects(areas.get(j))) {
//                     if(entry.key.toString().equals("27809914")) { println "area:$j"}
                     def areaAnnotations = annotationPerArea.get(j)
                     areaAnnotations.add(entry)
                     annotationPerArea.put(j,areaAnnotations)
//                     if(entry.key.toString().equals("27809914")) {println "area:${annotationPerArea.get(j)}"}
//                     break
                 }
             }
         }
     }
     areas.eachWithIndex { area, i ->
         println "Area $i=${annotationPerArea.get(i).size()}"
     }
     //annotationPerArea = [ id, [ {id,form} {id, form} ...   ] ]

     def listThread = []

     annotationPerArea.eachWithIndex { key, value, i->
         def lisEntry = value
         def th = Thread.start {
             //println "value=$value"
             value.eachWithIndex { current,indice ->
//                 if(key.toString().equals("6")) {
//                     println current.key
//                 }
                 def currentKey = current.key
                 def currentValue = current.value

                 if(indice%50==0) {
                     println "$i => $indice/${value.size()}"
                 }
                 def currentGeom = currentValue

                 if(currentKey.toString().equals("27809914")) {
                     println "*********************** 27809914"
                     println currentGeom.isValid()
                 }
                 if(currentKey.toString().equals("27808861")) {
                     println "*********************** 27808861"
                     println currentGeom.isValid()
                 }
                 //27808861)
                 if(currentGeom.isValid()) {
                     lisEntry.eachWithIndex { compared, j ->
//                         println "j=$j"
//                         println compared.key.toString()
                         if(currentKey.toString().equals("27808861") && compared.key.toString().equals("27809914")) {
                             println "check"
                             println Long.parseLong(currentKey.toString()) < Long.parseLong(compared.key.toString())
                         }

                          if(Long.parseLong(currentKey.toString()) < Long.parseLong(compared.key.toString())) {
                              def comparedGeom = compared.value
                              if(comparedGeom.isValid()) {
                                  def intersectGeom = comparedGeom.intersection(currentGeom)
                                  if(intersectGeom.getLength()>=minIntersectLength) {
                                      data << [compared.key,currentKey]
                                  }
                              }
                          }
                      }
                 }

             }
         }
         listThread << th
     }

     listThread.eachWithIndex { t,i->
         t.join()
         println "Thread $i is finished"
     }
     data
 }

/* * Split a list in number list
 * @param list List to split
 * @param number Number of sublist to create
 * @return List of list
 */
public ArrayList[] splitList(List list, long number) {
        //split indexCollection files list for each server
        ArrayList[] pictureByServer = new ArrayList[number];
        // initialize all your small ArrayList groups
        for (int i = 0; i < number; i++){
            pictureByServer[i] = new ArrayList();
        }

        double idx = 0d;
        double incrx = (double) ((double) number / (double) list.size());
        // put your results into those arrays
        for (int i = 0; i < list.size(); i++) {
            pictureByServer[(int) Math.floor(idx)].add(list.get(i));
            idx = idx + incrx;
        }
        return pictureByServer;
}
