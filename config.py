# change all paths to match your specific setup

training_config = {
    "learning_rate": 0.001,
    "batch_size": 2**8,
    "epochs": 50,
    "optimizer": "adam",
    "dropout_rate": 0.3,
    "clip" : 1,
    "out" : "out_pth"
}

model_config = {
    "enc_emb_dim" : 1024,
    "dec_emb_dim" : 1024,
    "enc_hid_dim" : 512,
    "dec_hid_dim" : 512,
    "enc_dropout" : 0.5,
    "dec_dropout" : 0.5,
    "src_pad_id" : 0,
}

data_config = {
    "dset": 'ml-25m',
    "data_root": "data_root_pth",
    "raw": "raw_pth",
    "processed": "processed_pth",
    "max_length": 200,
    "min_frequency": 0,
    "ml-25m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "filename": "ml-25m.zip",
        "raw_dest": "ml-25m_raw_dest_pth",
        "targets": ["ratings.csv"],
        "delimeter": ",",
        "columns": {
            "userid": "userId",
            "itemid": "movieId",
            "timestamp": "timestamp"}
    },
    "yelp": {
            "url": "https://yelp-dataset.s3.amazonaws.com/YDC22/yelp_dataset.tar",
            "filename": "yelp_dataset.tar",
            "raw_dest": "yelp_raw_dest_pth",
            "targets": ["yelp_academic_dataset_review.json"],
            "columns": {
                "userid": "user_id",
                "itemid": "business_id",
                "timestamp": "date"}
        },
    "4square": {
                "url": "https://drive.google.com/file/d/0BwrgZ-IdrTotZ0U0ZER2ejI3VVk/view?resourcekey=0-rlHp_JcRyFAxN7v5OAGldw",
                "filename": "dataset_TIST2015.zip",
                "raw_dest": "4square_raw_dest_pth",
                "targets": ["dataset_TIST2015_Checkins.txt"],
                "delimeter": "\t",
                "header": ["User ID", "Venue ID", "UTC time", "Timezone offset"],
                "columns": {
                    "userid": "User ID",
                    "itemid": "Venue ID",
                    "timestamp": "UTC time"}
            },
    "lastfm-dataset-1K": {
                    "url": "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz",
                    "filename": "lastfm-dataset-1K.tar.gz",
                    "raw_dest": "lastfm-dataset-1K_raw_dest_pth",
                    "targets": ["userid-timestamp-artid-artname-traid-traname.tsv"],
                    "delimeter": "\t",
                    "header": ["userid", "timestamp", "musicbrainz-artist-id", "artist-name",
                                    "musicbrainz-track-id", "track-name"],
                    "columns": {
                        "userid": "userid",
                        "itemid": "musicbrainz-track-id",
                        "timestamp": "timestamp"}
                }
}


hyperparameters = {
    "k" : 250,
    "gamma" : 1,
}