export const myViewConfig = {
  "version": "1.0.17",
  "name": "DLPFC Samples",
  "description": "Visium Spatial Gene Expression data from 10x Genomics (PsychENCODE)",
  "initStrategy": "auto",
  "datasets": [
    {
      "uid": "visium",
      "files": [
        {
          "fileType": "anndata.zarr",
          "url": "https://users.wenglab.org/kresgeb/psych_encode_spatialDLPFC/data/DLPFC_Br8667_mid_manual_alignment_all/data.h5ad.zarr",
          "coordinationValues": {
            "obsType": "spot",
            "featureType": "gene",
            "featureValueType": "expression"
          },
          "options": {
            "obsFeatureMatrix": {
              "path": "X",
              "initialFeatureFilterPath": "var/genes_of_interest"
            },
            "obsLocations": {
              "path": "obsm/spatial"
            },
            "obsSegmentations": {
              "path": "obsm/segmentations"
            },
            "obsEmbedding": [
              {
                "path": "obsm/X_umap",
                "embeddingType": "UMAP"
              },
              {
                "path": "obsm/X_pca",
                "embeddingType": "PCA"
              }
            ],
            "obsSets": [
              {
                "name": "BayesSpace (k=9)",
                "path": "obs/bayes_space_k=9"
              },
              {
                "name": "BayesSpace (k=16)",
                "path": "obs/bayes_space_k=16"
              },
              {
                "name": "BayesSpace (k=28)",
                "path": "obs/bayes_space_k=28"
              },
              {
                "name": "Leiden Clusters",
                "path": "obs/leiden"
              }
            ]
          }
        },
        {
          "fileType": "image.ome-zarr",
          "url": "https://users.wenglab.org/kresgeb/psych_encode_spatialDLPFC/data/DLPFC_Br8667_mid_manual_alignment_all/image.ome.zarr"
        }
      ]
    }
  ],
  "coordinationSpace": {
    "obsType": {
      "A": "spot"
    },
    "spatialSegmentationLayer": {
      "A": {
        "radius": 65,
        "stroked": true,
        "visible": true,
        "opacity": 1
      }
    },
    "spatialImageLayer": {
      "A": [
        {
          "type": "raster",
          "index": 0,
          "colormap": null,
          "transparentColor": null,
          "opacity": 1,
          "domainType": "Min/Max",
          "channels": [
            {
              "selection": {
                "c": 0
              },
              "color": [
                255,
                0,
                0
              ],
              "visible": true,
              "slider": [
                0,
                255
              ]
            },
            {
              "selection": {
                "c": 1
              },
              "color": [
                0,
                255,
                0
              ],
              "visible": true,
              "slider": [
                0,
                255
              ]
            },
            {
              "selection": {
                "c": 2
              },
              "color": [
                0,
                0,
                255
              ],
              "visible": true,
              "slider": [
                0,
                255
              ]
            }
          ]
        }
      ]
    },
    "obsSetColor": {
      "A": [
        {
          "path": [
            "BayesSpace (k=9)"
          ],
          "color": [
            0,
            0,
            0
          ]
        },
        {
          "path": [
            "BayesSpace (k=9)",
            "1"
          ],
          "color": [
            88,
            81,
            87
          ]
        },
        {
          "path": [
            "BayesSpace (k=9)",
            "2"
          ],
          "color": [
            228,
            224,
            226
          ]
        },
        {
          "path": [
            "BayesSpace (k=9)",
            "3"
          ],
          "color": [
            229,
            71,
            55
          ]
        },
        {
          "path": [
            "BayesSpace (k=9)",
            "4"
          ],
          "color": [
            239,
            34,
            244
          ]
        },
        {
          "path": [
            "BayesSpace (k=9)",
            "5"
          ],
          "color": [
            105,
            252,
            77
          ]
        },
        {
          "path": [
            "BayesSpace (k=9)",
            "6"
          ],
          "color": [
            83,
            120,
            248
          ]
        },
        {
          "path": [
            "BayesSpace (k=9)",
            "7"
          ],
          "color": [
            243,
            182,
            54
          ]
        },
        {
          "path": [
            "BayesSpace (k=9)",
            "8"
          ],
          "color": [
            163,
            37,
            102
          ]
        },
        {
          "path": [
            "BayesSpace (k=9)",
            "9"
          ],
          "color": [
            111,
            251,
            207
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)"
          ],
          "color": [
            0,
            0,
            0
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "1"
          ],
          "color": [
            88,
            81,
            87
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "2"
          ],
          "color": [
            228,
            224,
            226
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "3"
          ],
          "color": [
            229,
            71,
            55
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "4"
          ],
          "color": [
            239,
            34,
            244
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "5"
          ],
          "color": [
            105,
            252,
            77
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "6"
          ],
          "color": [
            83,
            120,
            248
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "7"
          ],
          "color": [
            243,
            182,
            54
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "8"
          ],
          "color": [
            163,
            37,
            102
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "9"
          ],
          "color": [
            111,
            251,
            207
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "10"
          ],
          "color": [
            147,
            174,
            50
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "11"
          ],
          "color": [
            106,
            210,
            252
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "12"
          ],
          "color": [
            217,
            159,
            251
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "13"
          ],
          "color": [
            162,
            1,
            248
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "14"
          ],
          "color": [
            237,
            166,
            161
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "15"
          ],
          "color": [
            63,
            85,
            152
          ]
        },
        {
          "path": [
            "BayesSpace (k=16)",
            "16"
          ],
          "color": [
            183,
            83,
            39
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)"
          ],
          "color": [
            0,
            0,
            0
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "1"
          ],
          "color": [
            88,
            81,
            87
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "2"
          ],
          "color": [
            228,
            224,
            226
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "3"
          ],
          "color": [
            229,
            71,
            55
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "4"
          ],
          "color": [
            239,
            34,
            244
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "5"
          ],
          "color": [
            105,
            252,
            77
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "6"
          ],
          "color": [
            83,
            120,
            248
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "7"
          ],
          "color": [
            243,
            182,
            54
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "8"
          ],
          "color": [
            163,
            37,
            102
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "9"
          ],
          "color": [
            111,
            251,
            207
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "10"
          ],
          "color": [
            147,
            174,
            50
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "11"
          ],
          "color": [
            106,
            210,
            252
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "12"
          ],
          "color": [
            217,
            159,
            251
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "13"
          ],
          "color": [
            162,
            1,
            248
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "14"
          ],
          "color": [
            237,
            166,
            161
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "15"
          ],
          "color": [
            63,
            85,
            152
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "16"
          ],
          "color": [
            183,
            83,
            39
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "17"
          ],
          "color": [
            58,
            129,
            88
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "18"
          ],
          "color": [
            73,
            29,
            8
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "19"
          ],
          "color": [
            167,
            31,
            159
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "20"
          ],
          "color": [
            245,
            233,
            71
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "21"
          ],
          "color": [
            14,
            38,
            45
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "22"
          ],
          "color": [
            233,
            56,
            134
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "23"
          ],
          "color": [
            234,
            58,
            189
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "24"
          ],
          "color": [
            243,
            229,
            165
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "25"
          ],
          "color": [
            185,
            120,
            164
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "26"
          ],
          "color": [
            117,
            32,
            178
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "27"
          ],
          "color": [
            181,
            245,
            58
          ]
        },
        {
          "path": [
            "BayesSpace (k=28)",
            "28"
          ],
          "color": [
            192,
            203,
            253
          ]
        }
      ]
    },
    "obsColorEncoding": {
      "A": "cellSetSelection",
      "B": "geneSelection"
    },
    "spatialZoom": {
      "A": -2.598
    },
    "spatialTargetX": {
      "A": 1008.88
    },
    "spatialTargetY": {
      "A": 1004.69
    },
    "featureSelection": {
      "A": [
        "MBP"
      ]
    }
  },
  "layout": [
    {
      "component": "spatial",
      "coordinationScopes": {
        "obsType": "A",
        "spatialImageLayer": "A",
        "spatialSegmentationLayer": "A",
        "spatialZoom": "A",
        "spatialTargetX": "A",
        "spatialTargetY": "A",
        "obsColorEncoding": "A",
        "obsSetColor": "A"
      },
      "x": 0,
      "y": 0,
      "w": 6,
      "h": 6
    },
    {
      "component": "spatial",
      "coordinationScopes": {
        "obsType": "A",
        "spatialImageLayer": "A",
        "spatialSegmentationLayer": "A",
        "spatialZoom": "A",
        "spatialTargetX": "A",
        "spatialTargetY": "A",
        "obsColorEncoding": "B",
        "featureSelection": "A"
      },
      "x": 6,
      "y": 0,
      "w": 6,
      "h": 6
    },
    {
      "component": "heatmap",
      "coordinationScopes": {
        "obsType": "A",
        "obsColorEncoding": "A",
        "obsSetColor": "A"
      },
      "props": {
        "transpose": true
      },
      "x": 6,
      "y": 6,
      "w": 6,
      "h": 6
    },
    {
      "component": "layerController",
      "coordinationScopes": {
        "obsType": "A",
        "spatialImageLayer": "A",
        "spatialSegmentationLayer": "A"
      },
      "props": {
        "disableChannelsIfRgbDetected": true
      },
      "x": 0,
      "y": 6,
      "w": 2,
      "h": 6
    },
    {
      "component": "obsSets",
      "coordinationScopes": {
        "obsType": "A",
        "obsColorEncoding": "A",
        "obsSetColor": "A"
      },
      "x": 2,
      "y": 6,
      "w": 2,
      "h": 6
    },
    {
      "component": "featureList",
      "coordinationScopes": {
        "obsType": "A",
        "obsColorEncoding": "B",
        "featureSelection": "A"
      },
      "x": 4,
      "y": 6,
      "w": 2,
      "h": 6
    }
  ]
}