import os
import streamlit as st
from streamlit_folium import folium_static
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import seaborn as sns
import Bibliotheques.bikebiblio as bike
import Bibliotheques.ReadAndCleanDataBike as dataCleaning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.linear_model import Lasso,SGDRegressor,ElasticNet,Ridge,LinearRegression
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


df = dataCleaning.LoadSaved()

st.write('''
![alt text][logo]
[logo]: https://static.lpnt.fr/images/2018/09/05/16794587lpw-16794961-article-velovoiturescameras-jpg_5541352_660x281.jpg "Logo Title Text 2"
''')

st.write('''
     # BIKE P(Y)TH
     
     ## Analyse du trafic cycliste à PARIS
     
     #### Etude allant de Janvier 2018 à Décembre 2019
     
     
''')


    
st.sidebar.header("Navigation")
rad = st.sidebar.radio("",['Projet','Datasets','Cartographie','Data Visualisations','Prédictions','Résultats'])

if rad == "Projet":
    
    html_temp = """
    <div style="background-color:tomato;color:white;font-size:30px"<p>L'Equipe PROJET</p><p>ENZEL Lola</p><p>BAYIHA Jean Arnaud</p><p>MAKNATT Mustapha</p></div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    st.write('''
    ## Problématique : 
         
    > Quels aménagements urbains ou nouvelles organisations peut-on mettre en place afin d’améliorer le trafic cyclable dans Paris ?
    ''')

    st.write('''     
    ### I. Situation
    ''')
    
    st.write('''
    > En 2015, Paris votait un grand plan destiné à transformer le paysage du trafic cyclable parisien.
    
    > Le but étant : "Pour que Paris devienne une capitale mondiale du vélo !".
    
    > Paris a d'ores et déjà amorcé un tournant pour favoriser la pratique cycliste.
    > Par exemple le Vélib’, qui a révolutionné l’usage du vélo dans Paris, est le
    > symbole de cette évolution. 
    
    > Aujourd'hui, il faut analyser les données de ces différents changements pour continuer à opérer aux bonnes améliorations, privilégier les 
    > bons emplacements d'investissements et choisir le bon type d'infrastructure.
    ''')
    
    st.write('''     
    ### II. Présentation de notre DATASET
    ''')
    
    st.write('''
    > Notre Dataset provient du site opendata Paris, voici le lien pour le télécharger : [Comptage vélo - Historique - Données compteurs](https://opendata.paris.fr/explore/dataset/comptage-velo-historique-donnees-compteurs/information/)
    
    > Ici un extrait de ce Dataset :
    ''')
    

    st.dataframe(df.head())
    
    st.write('''
    > Nous avions le choix entre 2 archives de données et nous avons opté pour le comptage horaire par compteur.
    
    > Voici les différentes colonnes de celui-ci :

    >*  Identifiant du compteur
    >*  Nom du compteur
    >*  Identifiant du site de comptage
    >*  Nom du site de comptage
    >*  Comptage horaire
    >*  Date et heure de comptage
    >*  Date d'installation du site de comptage
    >*  Lien vers photo du site de comptage
    >*  Coordonnées géographiques
    
    > Nous avons aussi choisit de fusionner les années de 2018 et 2019 en un seul et même Dataset, pour obtenir plus de données à étudier.
    ''')
    
if rad == "Datasets":
    st.write("Voici les différents dataset utilisés pour notre étude et sur lesquels nous avons récupérés les variables dont nous avions besoin. L'exploration de chaque dataset sera possible via différents outils.")
    
    def fileselector(folder_path='./Datasets'):
        filenames = os.listdir("./Datasets")
        selected_filename = st.selectbox("Sélectionner un fichier",filenames)
        return os.path.join(folder_path, selected_filename)
    
    
    filename = fileselector()
    st.info("Vous avez sélectionné {}".format(filename))
    
    #Read data
    df = pd.read_csv(filename, sep=';') 
    #Show data
    if st.checkbox("Montrer le Dataset"):
        number = st.number_input("Nombre de lignes choisies",5,100)
        st.dataframe(df.head(number))
    
    #Show columns
    if st.button("Noms des colonnes"):
        st.write(df.columns)
        
    #Show shape
    if st.checkbox("Dimension du Dataset"):
        st.write(df.shape)
        data_dim = st.radio("Montrer les Dimensions par",("Lignes","Colonnes"))
        if data_dim == "Lignes":
            st.text('Nombre de lignes')
            st.write(df.shape[0])
        elif data_dim == "Colonnes":
            st.text('Nombre de colonnes')
            st.write(df.shape[1])
            
    
    #Select Columns
    if st.checkbox("Sélectionner les colonnes à montrer"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)
        
        
        
    # Show datatypes
    #if st.button('Value Counts'):
    #    st.text("Value Counts By Target/Class")
     #   st.write(df.iloc[:,-1].value_counts())
        
    #Show Data types'):
    if st.button('Type de données du Dataset'):
        st.write(df.dtypes)
        
    #Show summary
    if st.checkbox("Description du Dataset"):
        st.write(df.describe())
        

    
    
if rad == "Cartographie":
    st.write("")
    st.write("Voici la carte des différents compteurs sur Paris :")    
    
    filenames = ["Emplacement des compteurs","Par arrondissement"]
    selected_vis = st.selectbox("Sélectionner une visualisation",filenames)
    if selected_vis == 'Par arrondissement' :        
        option = st.radio("",['Aucun filtre','Jour de grève','Vacances scolaires','Jour férié','Autre jour uniquement'])
        
        if option == "Jour de grève":
            df_filtred = df[df.jourdegreve == 1]
        elif option == "Vacances scolaires":
            df_filtred = df[df.vacances_zone_c == 1]
        elif option == "Autre jour uniquement":
            df_filtred = df[(df.holiday != 1)&(df.vacances_zone_c != 1)&(df.jourdegreve != 1)]
        elif option == "Jour férié":
            df_filtred = df[df.holiday == 1]
        else:
            df_filtred = df
        
        
        filenames = ["Tous","Par jour","Par heure","Par mois","Condition météo",]
        selected_vis = st.selectbox("Sélectionner une visualisation",filenames)
        mask = ""
        filenames = ["Tous"]
        
        if selected_vis != 'Par jour': 
            option = st.radio("",['Tous','WeekEnd uniqument','En semaine uniquement'])

            if option == "WeekEnd uniqument":
                df_filtred = df_filtred[(df_filtred.weekday == 5) | (df_filtred.weekday == 6)]
            elif option == "En semaine uniquement":
                df_filtred = df_filtred[(df_filtred.weekday != 5) & (df_filtred.weekday != 6)]

            
        
        if selected_vis == 'Par jour': 
            filenames = filenames + list(df_filtred.sort_values(by='weekday').jourdelasemaine.unique())
            selected_vis = st.selectbox("Sélectionner une valeur",filenames)
            mask = (df_filtred['jourdelasemaine']==selected_vis)             
        elif selected_vis == 'Par heure':   
            filenames = filenames + list(df_filtred.sort_values(by='hour').hour.unique())
            selected_vis = st.selectbox("Sélectionner une tranche horaire",filenames)
            st.write(selected_vis)
            mask = (df_filtred['hour']==selected_vis) 
        elif selected_vis == 'Par mois':     
            filenames = filenames + list(df_filtred.sort_values(by='month').mois.unique())
            selected_vis = st.selectbox("Sélectionner un mois",filenames)
            st.write(selected_vis)
            mask = (df_filtred['mois']==selected_vis) 
        elif selected_vis == 'Condition météo':                           
            filenames = filenames + list( df_filtred.meteo_cat.unique())
            selected_vis = st.selectbox("Sélectionner ",filenames)
            st.write(selected_vis)
            mask = (df_filtred['meteo_cat']==selected_vis) 
            
        
        df_grouped,map = bike.GetMapParArrondissement(df_filtred,mask,selected_vis!="Tous")
        if selected_vis!="Tous":
            st.write("Nombre de passage moyen par heure : ",df_filtred[mask].comptagehoraire.mean())
        else:
            st.write("Nombre de passage moyen par heure : ",df_filtred.comptagehoraire.mean())
            
        folium_static(map)
    if selected_vis == 'Emplacement des conmpteurs' :  
        st.map(df.rename(columns={ "long": "lon"}))
    
if rad == "Data Visualisations":
    st.write("A partir de notre dataframe nettoyé et corrigé, nous allons utiliser quelques visualisations pour mieux comprendre et analyser ces données. ")
    
    filenames = ["Correlation","Distribution par mois","Distribution par jour","Distribution par heure",'Vacances scolaires','Jours de grèves','Jours fériés']
    selected_vis = st.selectbox("Sélectionner une visualisation",filenames)          
            
    functions_to_apply = {
        'comptagehoraire' : lambda comptage: comptage.mean()
        }
              
    if selected_vis == 'Correlation' : 
        df_descritive = df.drop(columns=["nomcompteur", "idsite",'nomsite','dateinstall','photo','coord','year','day','weekofyear','lat','long'
                                    ,'arrondissement','jourdelasemaine','mois'])
        df_descritive.info()
        dfcor = df_descritive.corr()
        fig, ax = plt.subplots(figsize=(12,12))
        sns.heatmap(dfcor, annot= True, ax= ax, cmap="plasma");
        st.pyplot(fig)
        
    if selected_vis == 'Jours de grèves' :   
        df4 = df.groupby('jourdegreve',as_index = False).agg(functions_to_apply)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.barplot(x=df4.jourdegreve, y=df4["comptagehoraire"], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right"); 
        st.pyplot(fig)

    if selected_vis == 'Jours fériés' :  
        df4 = df.groupby('holiday',as_index = False).agg(functions_to_apply)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.barplot(x=df4.holiday, y=df4["comptagehoraire"], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right"); 
        st.pyplot(fig)

        
    if selected_vis == 'Vacances scolaires' : 
        df4 = df.groupby('vacances_zone_c',as_index = False).agg(functions_to_apply)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.barplot(x=df4.vacances_zone_c, y=df4["comptagehoraire"], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right"); 
        st.pyplot(fig)

    if selected_vis == 'Distribution par mois' :         
        functions_to_apply = {
            'comptagehoraire' : lambda comptage: comptage.mean(),
            'month' : lambda month: month.mean()
            }

        df4 = df.groupby('mois',as_index = False).agg(functions_to_apply)
        df4 = df4.sort_values(by='month', ascending=True)

        #st.dataframe(df4)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.barplot(x=df4.mois, y=df4["comptagehoraire"], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right");
        st.pyplot(fig)
    
    
    if selected_vis == 'Distribution par jour' :   
        functions_to_apply = {
        'comptagehoraire' : lambda comptage: comptage.mean(),
        'weekday' : lambda weekday: weekday.mean()
        }

        dfparjour = df.groupby('jourdelasemaine',as_index = False).agg(functions_to_apply)
        dfparjour = dfparjour.sort_values(by='weekday', ascending=True)
        dfparjour.reset_index()

        fig, ax = plt.subplots(figsize=(10,10))
        sns.barplot(x=dfparjour.jourdelasemaine, y=dfparjour["comptagehoraire"], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right");
        plt.xlabel("")
        plt.ylabel("Nombre moyen de passage par heure")


        st.pyplot(fig)
    
    if selected_vis == 'Distribution par heure' :  
        functions_to_apply = {
        'comptagehoraire' : lambda comptage: comptage.mean()
        }

        df4 = df.groupby('hour',as_index = False).agg(functions_to_apply)
        df4 = df4.sort_values(by='hour', ascending=True)

        fig, ax = plt.subplots(figsize=(10,10))
        sns.barplot(x=df4.hour, y=df4["comptagehoraire"], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right");

        st.pyplot(fig)


    
if rad == "Prédictions":
    
    st.write('''
    ## Prédictions
    ''')
    
    st.write("Voici la présentation de quelques modèles de prédictions et leurs résultats : ")
    
    st.write("Nous prendrons ici le dataset finalisé avec toutes les nouvelles variables ajoutées (voyez ici un échantillon).")
    
    df = dataCleaning.LoadSaved()
                       
    st.dataframe(df.head())
    
    st.header("Linear Regression")
    
    y=df.comptagehoraire
    X=df[['year','day','weekday','weekofyear','month','hour','jourdegreve','meteo','holiday','vacances_zone_c','arrondissement']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lrgr = LinearRegression()
    lrgr.fit(X_train,y_train)
    pred = lrgr.predict(X_test)
    
    score = lrgr.score(X_train, y_train)
    mse = mean_squared_error(y_test,pred)
    rmse = sqrt(mse)

    st.markdown(f"""
    Linear Regression model trained :
    - MSE : {mse}
    - RMSE : {rmse}
    - SCORE : {score}
    """)
    
    st.header("Ridge Regressor")
    
    encoder=OneHotEncoder()
    features_enc=encoder.fit_transform(X)
    
    X_train,X_test , y_train,y_test = train_test_split(features_enc,y,test_size=0.2)
    
    model_ridge=Ridge(max_iter=10)
    param_grid_ridge  = {'solver': ['auto', 'svd', 'cholesky'],'alpha': np.linspace(0,10,15)}
    
    grid_ridge= GridSearchCV(model_ridge,
                         param_grid=param_grid_ridge,
                         cv=5,
                         scoring='r2')
    grid_ridge.fit(X_train,y_train)
    
    best_param = grid_ridge.best_params_
    best_score = grid_ridge.best_score_
    
    st.markdown(f"""
    Ridge Regressor model trained :
    - Best Params : {best_param}
    - Best SCORE : {best_score}
    """)
    
    st.header("SGDRegressor")
    
    encoder=OneHotEncoder()
    features_enc=encoder.fit_transform(X)
    
    X_train,X_test , y_train,y_test = train_test_split(features_enc,y,test_size=0.2)
    
    model_sgd= SGDRegressor(max_iter=10)
    param_grid_sgd = {'learning_rate': ['constant','optimal'], 
                      'penalty':    ['l2', 'l1'],
                      'alpha':      np.linspace(0.0001,1,10),
                      'loss':       ['squared_loss', 'huber'] }
    
    grid_sgd=GridSearchCV(model_sgd,
                         param_grid=param_grid_sgd,
                         cv=5,
                         scoring='r2')
    grid_sgd.fit(X_train,y_train)
    
    best_param_sgd = grid_sgd.best_params_
    best_score_sgd = grid_sgd.best_score_
    
    st.markdown(f"""
    SGDRegressor model trained :
    - Best Params : {best_param_sgd}
    - Best SCORE : {best_score_sgd}
    """)
    
    st.header("Random Forest")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf_rf  = RandomForestClassifier()
    param_grid_rf  = [{'n_estimators': [10, 100],
                         'min_samples_leaf': [1],
                         'max_features': ['sqrt']}]
    gridcv = GridSearchCV(clf_rf,
                           param_grid=param_grid_rf,
                           scoring='accuracy',
                           cv=2)
    gridcv.fit(X_train, y_train)
    
    best_param_rf = grid_cv.best_params_
    best_score_rf = grid_cv.best_score_
    
    st.markdown(f"""
    Random Forest model trained :
    - Best Params : {best_param_rf}
    - Best SCORE : {best_score_rf}
    """)
    
    
if rad == "Résultats":
    st.write('''
    ## Résultats
    
    > Ce projet nous a permis d’observer la répartition globale du trafic cyclable dans la capitale :
    >* Grâce au découpage par mois nous observons une hausse du trafic en septembre et en décembre.
    >* Grâce au découpage par jour de la semaine nous constatons que le trafic est plus dense la semaine que le week-end.
    >* Grâce au découpage par tranches horaires nous remarquons des pics de trafic durant les heures de pointes.
    >* Grâce au découpage par jours fériés, vacances et jour de grève nous pouvons voir que le trafic cyclable moyen est en baisse les jours fériés ou les jours de grève.
    >* Grâce au découpage par quartiers nous nous apercevons que les arrondissement 3,10 et 11 sont les plus fréquentés par les cyclistes.

    > L’objectif principal qui était d’avoir une visibilité sur le trafic dans les zones de Paris a été atteint. En effet, grâce à nos  observations nous pouvons voir les zones les plus fréquentées de Paris.
    > Nous pouvons également connaitre les habitudes des cyclistes, tout comme pour le trafic automobile, les fréquences d’affluences correspondent aux heures de pointes.

    > En outre, nous n’avons pas était en mesure de trouver un modèle satisfaisant pour prédire le trafic. En effet, le problème sanitaire actuel a totalement bouleversé les habitudes et les données ne permettent donc pas la mise en place d’un modèle efficace pour le moment.
    
    ''')
    
