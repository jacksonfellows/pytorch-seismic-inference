CREATE TABLE picks (
       -- Station metadata
       network TEXT,
       station TEXT,
       location TEXT,
       channel TEXT,
       -- Pick info
       source_type TEXT,
       pick_time DATETIME,
       pick_prob REAL
);
