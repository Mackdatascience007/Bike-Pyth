#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import Ridge, LassoCV, Lasso
from sklearn.ensemble import RandomForestClassifier
import folium
from shapely.geometry import shape, Point
import os
import json,geojsonio
from zipfile import ZipFile


def extractDatas():    
    if checkFileExistance('Datasets/Bike P(Y)TH - PARIS.csv') == False :
        test_file_name = "Datasets/Bike P(Y)TH - PARIS.zip"
        with ZipFile(test_file_name, 'r') as zip:
            zip.extractall("Datasets") 
        
        
    if checkFileExistance("Datasets/dfsaved.csv")  == False :
        test_file_name = "Datasets/dfsaved.zip"
        with ZipFile(test_file_name, 'r') as zip:
            zip.extractall("Datasets") 
            
def setJF(df,row,df_jour_feries):    
    mask = (row.date == df['date'])
    df.loc[mask, 'holiday'] = 1


def setJourDeGreve(df,row):    
    mask = (row.date_de_debut <= df['date']) & (df['date'] <= row.date_de_fin)
    df.loc[mask, 'jourdegreve'] = 1




def readAndClean(nomCsv,separateur):
    extractDatas()
    # Lecture du dataset 
    df = pd.read_csv(nomCsv,sep=separateur)

    # Nous constatons à ce stade que le type de variable n'est pas le bon pour chaque colonne il va donc
    # transformer ces variables pour pouvoir les analyser correctement
    # Par ailleurs, nous constatons aussi que le nom des colonnes devra être revu pour harmoniser l'ensemble
    # et faciliter leurs utilisations


    #Renommage des colonnes
    df.set_axis(['idcompteur', 'nomcompteur', 'idsite', 'nomsite', 'comptagehoraire', 'dateheurecomptage', 'dateinstall', 'photo', 'coord'], axis='columns', inplace=True)

    # Changement du type de la colonne comptageHoraire
    df['comptagehoraire'] = df['comptagehoraire'].apply(pd.to_numeric, errors='coerce')

    # Changement du type de la colonne idsite
    df['idsite'] = df['idsite'].apply(pd.to_numeric, errors='coerce')

    # Création des colonnes liées à la date et heure
    
    df['date'] = pd.to_datetime(df.dateheurecomptage,errors = 'coerce',utc = True) 
    
    df['datemois'] = df["date"].dt.to_period('M')
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['weekofyear'] = df['date'].dt.isocalendar().week    
    df['month'] = df['date'].dt.month
    df['hour'] = df.date.dt.hour
    df['mois'] = df['month'].replace(list(range(1,13)), [' Janvier ' , ' Février ' , ' Mars ' , ' Avril ' , ' Mai ' , ' Juin ' , ' Juillet ' 
                                                         , ' Août ' , ' Septembre ' , ' Octobre ' , ' Novembre ' , ' Décembre ' ])
    df['jourdelasemaine'] = df['weekday'].replace(list(range(0,7)), ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche'])
    #Suppression de la ligne de titre en trop
    indexNames = df[df.idcompteur == "Identifiant du compteur"].index
    df.drop(indexNames , inplace=True)
    #Variable numérique pour l'identifiant compteur
    df["numidcompteur"] = df["idcompteur"].replace(df["idcompteur"].unique(), list(range(0,len(df["idcompteur"].unique()))))
    df['dateheurecomptage'] = pd.to_datetime(df.dateheurecomptage,errors = 'coerce',utc = True) 

    # Suppression des lignes ne possédant pas de coordonnées (la dernière ligne du fichier est corrompue)
    df = df[-df['coord'].isnull()]

    # Passage de type "object" à "datetime" pour la variable datainstall

    df.loc[df['dateinstall']=="Date d'installation du site de comptage",'dateinstall'] ='06/09/2019'
    df['dateinstall'] = pd.to_datetime(df['dateinstall'])

    # A ce stade il n'y a plus aucune valeur nulle ou de doublons et nous avons harmonisé le nom des variables ainsi que leur type

    # Splitter la colonne 'coord' en 2 colonnes comprenant les 'longitudes' et les 'latitudes'
    # Créer 2 listes pour la boucle for
    lat = []
    lon = []

    # Pour chaque ligne de la variables 'coord',
    for row in df['coord']:
        # On va,
        try:
            # Splitter la ligne avec la virgule et utiliser 
            # append pour ce qui est avant celle-ci à ajouter dans 'lat'
            lat.append(row.split(',')[0])
            # Splitter la ligne avec la virgule et utiliser 
            # append pour ce qui est avant celle-ci à ajouter dans 'lon'
            lon.append(row.split(',')[1])
        # Si il y a une erreur de coordonnées on crée une exception
        except:
            lat.append(np.NaN)
            lon.append(np.NaN)

    # Création des 2 nouvelles colonnes avec lat et lon
    df['lat'] = lat
    df['long'] = lon

    # Transformation du type de variable des colonnes latitude et longitude
    df['lat'] = df['lat'].astype(float)
    df['long'] = df['long'].astype(float)
    #chargement du dataset des jour de grèves
    df_mouv = pd.read_csv('Datasets//mouvements-sociaux-depuis-2002.csv',sep=';')
    #suppression des colonnes inutiles
    df_mouv = df_mouv.drop(['motif','organisations_syndicales','metiers_cibles','population','nombre_grevistes'], axis=1)
    #si pas de date de fin, on copie la date de début
    df_mouv.date_de_fin = df_mouv.date_de_fin.fillna(df_mouv.date_de_debut)
    df_mouv.taux_grevistes = df_mouv.taux_grevistes.fillna(df_mouv.taux_grevistes.mean())
    df_mouv.date_de_fin = pd.to_datetime(df_mouv.date_de_fin,errors = 'coerce',utc = True) 
    df_mouv.date_de_debut = pd.to_datetime(df_mouv.date_de_debut,errors = 'coerce',utc = True) 

    df["jourdegreve"] = 0
    # pour chaque grève, on recherche dans df les lignes correspondantes
    for i,row in enumerate(df_mouv.T):
        setJourDeGreve(df,df_mouv.T[i])
    
    #chargement du dataset pour la météo
    df_meteo_all = pd.read_csv('Datasets//donnees-synop-essentielles-omm.csv',sep=';')
    df_meteo = df_meteo_all[df_meteo_all['communes (name)'] =="Athis-Mons"]
    df_meteo['date'] = pd.to_datetime(df_meteo.Date,errors = 'coerce',utc = True) 
    
    
    df_meteo = df_meteo[df_meteo['date'].dt.year < 2020]
    df_meteo = df_meteo[df_meteo['date'].dt.year > 2017]
    df_saved=df
    df_meteo['Temps présent'] = pd.to_numeric(df_meteo['Temps présent'])
    colonneagarder=['date','Temps présent']
    df_meteo = df_meteo[colonneagarder]

    df_meteo = df_meteo.rename(columns={ "Temps présent": "meteo"})

    df_meteo['meteo_cat'] = pd.cut( df_meteo['meteo'],[0,49,69,79,84,86,94,99], labels=["Dégagé", "Pluie", "Neige","Pluie","Neige","Pluie","Orage"],ordered=False)


    df_meteo['meteo'] = pd.cut( df_meteo['meteo'],[0,49,69,79,84,86,94,99], labels=[0, 1, 2,1,2,1,3],ordered=False)

    df = df.merge(df_meteo,how = 'left',left_on = 'date',right_on = 'date')

    df['meteo_cat'] = df['meteo_cat'].fillna("Dégagé")
    df['meteo'] = df['meteo'].fillna(0)
    

        
    # Ajout des jours fériés
    df_jour_feries = pd.read_csv('Datasets//jours_feries_metropole.csv')

    df_jour_feries['date'] = pd.to_datetime(df_jour_feries["date"],errors = 'coerce',utc = True)
    df_jour_feries['date'] = pd.to_datetime(df_jour_feries["date"].dt.strftime('%Y-%m-%d'))

    df['date'] = pd.to_datetime(df["date"].dt.strftime('%Y-%m-%d'))

    df['holiday'] = 0

    for i,row in enumerate(df_jour_feries.T):
            setJF(df,df_jour_feries.T[i],df_jour_feries)

    # Gestion des vacances
    df_vacances = pd.read_csv('Datasets//data-vacances.csv', sep=',')

    df_vacances = df_vacances.drop(['vacances_zone_a','vacances_zone_b','nom_vacances'],axis=1)

    ZoneC = df_vacances[ df_vacances['vacances_zone_c'] == False ].index
    df_vacances.drop(ZoneC , inplace=True)

    dates = (df_vacances['date'] >= '2018-01-01') & (df_vacances['date'] <= '2019-12-31')
    df_vacances = df_vacances.loc[dates]

    df_vacances['date'] = pd.to_datetime(df_vacances["date"],errors = 'coerce',utc = True)

    df_vacances['date'] = df_vacances["date"].dt.strftime('%Y-%m-%d')
    df['date'] = df["date"].dt.strftime('%Y-%m-%d')
    df=df.merge(right = df_vacances, on = 'date' , how = 'left')
    df['vacances_zone_c'] = df['vacances_zone_c'].fillna(0)
    df['vacances_zone_c'] = df['vacances_zone_c'].replace(True, 1)

    df['arrondissement'] = 255
    for numidcompteur in df.numidcompteur.unique():    
        long = df[df['numidcompteur'] == numidcompteur].head(1).long
        lat = df[df['numidcompteur'] == numidcompteur].head(1).lat
        if GetArrondissement(long,lat) != None :
            df.loc[df['numidcompteur'] == numidcompteur,'arrondissement'] = int(GetArrondissement(long,lat))
    df.to_csv("Datasets//dfsaved.csv")
    return df
def checkFileExistance(filePath):
    try:
        with open(filePath, 'r') as f:
            return True
    except FileNotFoundError as e:
        return False
    except IOError as e:
        return False
def LoadSaved():
    extractDatas()
    if checkFileExistance('Datasets/dfsaved.csv') :
        df=pd.read_csv("Datasets//dfsaved.csv")
    else:
        df = readAndClean('Datasets/Bike P(Y)TH - PARIS.csv',';')        
    return df

def GetArrondissement(long,lat):
    # construct point based on lon/lat returned by geocoder
    point = Point(long,lat)
    # load GeoJSON file containing sectors
    with open("datas//arrondissements.geojson") as f:
        js = json.load(f)
    # check each polygon to see if it contains the point
    for feature in js['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):            
            if feature['properties']['c_ar'] == None :
                return 255
            else:
                return feature['properties']['c_ar']
            

    