# TFG-Coixi-IoT-public

Aquest repositori conté la versió documental del codi font de la interfície web i del servidor IoT desenvolupats en el Treball de Fi de Grau "Coixí sensoritzat per al monitoratge de la distribució de pressió en cadira de rodes".

El sistema permet rebre les dades enviades pel prototip, visualitzar el mapa de pressió i consultar la informació associada a les sessions de monitoratge.

Per motius de seguretat, aquest repositori no inclou credencials, certificats digitals, claus privades ni dades sensibles d'AWS. Els paràmetres de configuració s'han substituït per camps genèrics o fitxers d'exemple.

## Fitxers principals

app.py: aplicació principal de la interfície web i servidor.
requirements.txt: dependències necessàries per executar el projecte.
config.example.py: exemple de configuració sense credencials reals.

## Execució

Instal·lar les dependències:

pip install -r requirements.txt

Executar l'aplicació:

python app.py
