
import folium
import json
import shapely.geometry
from shapely.geometry import shape, Point
import os
import json,geojsonio

 
def GetColor(nb_passage,nb_max):
    tier_max = nb_max/3
    if(tier_max > nb_passage):
        return '#ee114d'
    if( tier_max < nb_passage < (tier_max*2)):        
        return '#eed311'
    if((tier_max*2) < nb_passage  ):        
        return '#28c0e2'
    
# Retourne une map Folium du nombre de passage horaire moyen par compteur représenté par des cercles, avec le mask passé en argument
def GetMap(df, mask,utiliserMask):
    # On cherche à obtenir une df avec le comptage horaire moyen, et les coordonnées GPS
    functions_to_apply = {
        'comptagehoraire' : lambda comptage: comptage.sum(),
        'lat' : lambda lat:lat.max(),
        'long' : lambda long:long.max()    }
    # Si on veut utiliser le mask on l'applique sinon on regroupe df 
    if utiliserMask:
        df_grouped = df[mask].groupby('idcompteur').agg(functions_to_apply)
    else:
        df_grouped = df.groupby('idcompteur').agg(functions_to_apply)
        
    
    df_grouped = df_grouped.dropna()
    
    # pour savoir la couleur et le rayon des cercles, on a besoin de connaitre le nombre max 
    nb_max = df_grouped.comptagehoraire.max()

    coords = (48.856614,2.3522219)
    map = folium.Map(width=550,height=350,location=coords, tiles='OpenStreetMap', zoom_start=12)
    # pour chaque ligne du df regroupé par compteur, on créé un cercle
    for ligne in df_grouped.T:
        #Le rayon est prportionelle au nombre de passage 
        t_radius = df_grouped.T[ligne].comptagehoraire/nb_max*50
        folium.CircleMarker(
           location = (df_grouped.T[ligne].lat, df_grouped.T[ligne].long),
           radius = t_radius,
           color = GetColor(df_grouped.T[ligne].comptagehoraire,nb_max),
           fill = True,
           fill_color = GetColor(df_grouped.T[ligne].comptagehoraire,nb_max),
           popup = str(df_grouped.T[ligne].comptagehoraire)
        ).add_to(map)
        coordonnees_gps = [df_grouped.T[ligne].lat, df_grouped.T[ligne].long]
    return df_grouped,map

    
# Retourne une map Folium du nombre de passage horaire moyen par arrondissement, avec le mask passé en argument
def GetMapParArrondissement(df, mask,utiliserMask):
    #on cherche à obtenir un df avec le nombre de passage moyen par arrondissement
    functions_to_apply = {
        'comptagehoraire' : lambda comptage: comptage.mean()
        }
    # Si on veut utiliser le mask on l'applique sinon on regroupe df 
    if utiliserMask:
        df_masked = df[mask]
    else:
        df_masked = df
    df_grouped = df_masked.groupby('arrondissement',as_index = False).agg(functions_to_apply)
    
    df_grouped = df_grouped.dropna()
    df_grouped.drop(df_grouped.loc[df_grouped['arrondissement']==255].index)
    
    # On charge le geojson des arrondissement de paris
    with open("datas/arrondissements.geojson") as f:
        js = json.load(f)
        # Pour chaque arrondissement, on récupère la valeur moyenne du nombre de passage 
    for feature in js['features']:
        df2= df_masked[df_masked['arrondissement'] == feature['properties']['c_ar']]
        feature['properties']['Nbr_Passage'] = df2.comptagehoraire.mean() 
        df2= df[df['arrondissement'] == feature['properties']['c_ar']]
        feature['properties']['Nbr_Compteur'] = len(df2.numidcompteur.unique())
    # Création d'une map centrée sur le centre de Paris
    m = folium.Map(width=550,height=350,location = [48.856578, 2.351828], zoom_start = 11)
    # Pour une cohérence de tous les graphique, on définit l'échelle à partir du df complet 
    myscale = (df['comptagehoraire'].quantile((0,0.125,0.25,0.375,0.50,0.625,0.75,0.875,1))).tolist()
    
    # La valeur maximum est toujours présente dans le df, pour conserver la même échelle
    df_grouped=df_grouped.append({'arrondissement' : '255' , 'comptagehoraire' : df['comptagehoraire'].max()} , ignore_index=True)
    # Délimitation des arrondissement sur la carte de paris, 
    # en intégrant le nombre de passage moyen 
    folium.Choropleth(
       geo_data='datas/arrondissements.geojson', key_on = "feature.properties.c_ar",
                     data = df_grouped, columns = ["arrondissement", "comptagehoraire"],
                 fill_color = "YlOrRd",color_continuous_scale="Viridis",
                   nan_fill_color="black",highlight=True,threshold_scale = myscale
    ).add_to(m)
    
    style_function = lambda x: {'fillColor': '#ffffff', 
                                'color':'#000000', 
                                'fillOpacity': 0.1, 
                                'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 
                                    'color':'#000000', 
                                    'fillOpacity': 0.50, 
                                    'weight': 0.1}
    # Ajout des tooltip avec le nombre de passage et le nom de l'arrondissement
    NIL = folium.features.GeoJson(
    js,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function,
    tooltip=folium.features.GeoJsonTooltip(
        fields=['l_ar','Nbr_Passage','Nbr_Compteur'],
        aliases=['Nom : ','Nombre de passages : ','Nombre de compteur']
        )
    )
    m.add_child(NIL)
    m.keep_in_front(NIL)
    folium.LayerControl().add_to(m)
    
    return df_grouped,m