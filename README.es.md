# musicverse

[English](README.md) | [Español](README.es.md)


## ¿Qué es?

Un mapa 3D de grabaciones de música clásica donde la distancia significa algo: dos
grabaciones quedan cerca si suenan parecido.

Cada grabación pasa por un modelo de embeddings de audio que la convierte en un vector
de 512 dimensiones, y esos vectores después se comprimen a tres para poder recorrer el
resultado como si fuera un mapa estelar.

**En qué anda esto ahora:** el pipeline funciona de punta a punta. El frontend 3D
todavía no existe. Así que lo que hay acá es la maquinaria que construye el mapa, más
un montón de mediciones que hice para ver si el mapa significa algo, lo cual terminó
siendo lo más interesante.

## ¿Por qué?

Un catálogo de música clásica se puede recorrer por compositor, por año o por
título. No se puede recorrer por cómo suena.

Quería un lugar donde las obras estuvieran acomodadas por parecido sonoro y se
pudieran explorar como un universo: acercarte a una zona, ver qué hay alrededor,
y encontrar cosas que nunca habrías buscado por nombre.

## Lo que encontré hasta ahora

Los números salen de una corrida real sobre 46 grabaciones de la colección 78rpm del
Internet Archive. Es un corpus chico, así que no lo tomes tan literal.

### Mapea timbre, no obra

Acá tuve suerte. En el corpus cayeron dos grabaciones del Clair de Lune de Debussy:
una de Stokowski con la Orquesta de Filadelfia y otra de Moura Lympany al piano solo.

No se eligieron entre sí. Lo más cercano a Stokowski resultaron las Variaciones
Enigma de Elgar. Lo más cercano a Lympany, un estudio de Chopin tocado por Cortot.
Las dos en 0.863.

Misma obra, rincones opuestos del espacio, separadas por el instrumento que la tocó. El
patrón está por todo el corpus, un aria de Mozart emparejada con una pieza coral de
Rachmaninoff, dos cosas sin nada en común salvo una voz solista con acompañamiento.

Entonces este mapa tiene zonas de piano, de orquesta, de violín, de voz. No tiene una
zona de Debussy. Conviene saberlo antes de leer demasiado en un cluster.

### Las transferencias ruidosas se amontonan

Cada track lleva un `noise_score`, que es planitud espectral: 0 es tonal, 1 es ruido
blanco. Si partís el corpus por la mediana, las dos mitades no se comportan igual
(números de la corrida sin centrar, donde la escala se lee más fácil):

| | similitud coseno promedio |
|---|---|
| dentro de la mitad ruidosa | 0.836 |
| dentro de la mitad limpia | 0.773 |
| entre las dos mitades | 0.752 |

La mitad limpia está apenas por encima de la línea base del cruce. La ruidosa está muy
por encima. O sea que no son dos clusters simétricos: son las grabaciones ruidosas
cayéndose unas sobre otras, lo cual tiene sentido, porque cuando el ruido domina el
espectro queda menos música para distinguirlas.

Los siete discos Edison de la era acústica del corpus cayeron todos en la mitad alta.

![46 grabaciones proyectadas a 3D, coloreadas por noise_score](docs/img/preview.png)

Los puntos claros son las transferencias ruidosas. Se agrupan en una zona mientras las
limpias se reparten por todo el volumen.

### Las etiquetas automáticas no funcionaron

CLAP mete audio y texto en el mismo espacio, así que en teoría podés etiquetar una
grabación comparándola contra frases como "a solo piano piece". En la práctica la
etiqueta ganadora cambia según cómo escribas la frase. Probé cuatro redacciones de la
misma lista de instrumentos. Con una de ellas, 30 de 46 tracks salieron como "a full
symphony orchestra", incluido un estudio de Chopin al piano solo. Un preludio de órgano
de Bach lo llamó piano, orquesta u órgano según cómo lo escriba.

Los vectores de audio aguantan bien. Lo que se cae es la alineación con texto sobre
grabaciones de hace un siglo. Guardo el puntaje contra todas las etiquetas en vez de
solo la ganadora, así se puede revisar sin volver a procesar el audio.

### La proyección es lo bastante fiel

La *trustworthiness* mide cuánto de la vecindad en 512 dimensiones sobrevive el pasaje
a 3. Así como está, da 0.885. Centrando los vectores primero, sube a **0.955**. Misma
estructura, pero repartida sobre un rango de distancias más amplio, que es más fácil de
preservar en tres dimensiones. La lista de pares más similares casi no cambia entre las
dos versiones, por lo cual entiendo que centrar estira el espacio en vez de reorganizarlo.

## Cómo funciona

```
Internet Archive 78rpm  ->  fetch_audio.py  ->  data/raw/*.mp3
                                                data/manifest.json
                                     |
                                     v
      embeddings CLAP      ->  embed.py     ->  data/embeddings.npz
   (512-d, ventanas de 10s,                     un vector por track
    promediadas)                                + ruido + puntajes
                                     |
                                     v
        UMAP a 3D          ->  project.py   ->  docs/data/universe.json
                                                coordenadas, metadata,
                                                vecinos precalculados
                                     |
                                     v
                              frontend (todavía no :P)
```

Todo lo caro pasa offline. El sitio publicado es estático: carga un JSON y lo dibuja.
Sin servidor y sin base de datos.

Dos decisiones que tomé:

**Sin base de datos vectorial.** A esta escala sería infraestructura porque sí. Los
vecinos se calculan offline y viajan dentro del JSON. Eso cambia si el corpus llega a
seis cifras o si agrego búsqueda por audio en vivo.

**Los vecinos salen del espacio de 512 dimensiones, no del dibujo en 3D.** Las
proyecciones distorsionan las distancias más allá de la vecindad inmediata, así que
"grabaciones parecidas" en la interfaz no se puede leer del mapa.

## Estructura

```
pipeline/          Python, corre offline
  fetch_audio.py     descarga desde el Internet Archive
  embed.py           audio -> vectores CLAP
  project.py         vectores -> coordenadas 3D, más diagnósticos
docs/              el sitio estático publicado (GitHub Pages sirve esta carpeta)
  data/
    universe.json    el mapa
  img/
    preview.png      la proyección, para el README
data/              archivos de trabajo locales, fuera del repo
  raw/               audio descargado
  manifest.json      qué se descargó
  embeddings.npz     los vectores
```

El audio nunca entra al repositorio. Git maneja mal los binarios grandes y los deja en
el historial para siempre.

`docs/` guarda el frontend en vez de documentación porque GitHub Pages solo publica
desde la raíz del repo o desde una carpeta con ese nombre exacto.

## Cómo correrlo

Necesita Python 3.10+, ffmpeg, y de ser posible una GPU CUDA para la etapa de
embeddings. En CPU anda, pero tarda una vida.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r pipeline/requirements.txt

python pipeline/fetch_audio.py --per-composer 5   # descargar
python pipeline/embed.py                          # embeddings
python pipeline/project.py --center --plot        # proyección + diagnósticos
```

Cada etapa saltea lo que ya hizo, así que volver a correr sale barato. Los resultados de
búsqueda salen en orden aleatorio, así que correr el descargador de nuevo suma variedad
en vez de traer las mismas grabaciones.

`project.py` imprime diagnósticos en vez de solo generar un dibujo. UMAP siempre te va a
dibujar algo. Los números son los que te dicen si significa algo. `--plot` escribe la
imagen que ves arriba en `data/preview.png`.

## Datos, y una nota de copyright

El audio viene de la colección 78rpm del Internet Archive. La metadata de ahí es
despareja de maneras que me complicaron:

- El campo `creator` mezcla compositores e intérpretes sin distinguirlos. Un disco está
  acreditado a Rachmaninoff y a Chopin: es Rachmaninoff tocando Chopin. Así que la
  atribución de compositor acá es un artefacto de cómo busqué, no un dato, no la puedo
  usar como etiqueta.
- Un item es un lado de un disco, 4 min aprox. Las obras largas quedan partidas
  en varios items. Un track acá no equivale a una obra.

Sobre copyright: según la Music Modernization Act, las grabaciones estadounidenses
publicadas hasta 1925 son de dominio público, y la línea avanza un año cada enero. Buena
parte de la colección 78rpm es posterior a eso.

**Este repositorio no redistribuye audio.** Contiene coordenadas derivadas, metadata y
enlaces de vuelta a los items originales del Internet Archive. Lo que se descarga
localmente se queda local.

El código es MIT. Eso cubre solo el código.

## Roadmap (estimado)

- [x] Descargar grabaciones de dominio público, con reanudación
- [x] Embeddings CLAP con un puntaje de ruido por track
- [x] Proyección UMAP con diagnósticos de calidad
- [ ] Frontend en Three.js: recorrerlo, clickear un punto, ver sus vecinos
- [ ] Pasar de 150 grabaciones para que UMAP tenga más material
- [ ] Cruzar con MusicBrainz para tener compositor y período de verdad
- [ ] Colorear y filtrar por ruido, para que el artefacto se vea en vez de esconderse
- [ ] Reproducir un fragmento al clickear, limitado a lo que esté claramente en dominio
      público
- [ ] Melodía y armonía desde MIDI o transcripción automática. CLAP no representa altura
      de forma explícita y más datos no lo va a arreglar
