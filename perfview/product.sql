PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE files (
	id INTEGER NOT NULL, 
	virtual_folder VARCHAR, 
	name VARCHAR, 
	description VARCHAR, 
	signature VARCHAR,
	size_bytes BIGINT, 
	create_time DATETIME, 
	update_time DATETIME, 
	PRIMARY KEY (id)
);

CREATE INDEX ix_files_id ON files (id);
CREATE INDEX ix_files_name ON files (name);
CREATE INDEX ix_files_signature ON files (signature);
CREATE INDEX ix_files_virtual_folder ON files (virtual_folder);

CREATE TABLE views (
	id INTEGER NOT NULL, 
	virtual_folder VARCHAR, 
	name VARCHAR, 
	view_type VARCHAR, 
	meta_data VARCHAR, 
	create_time DATETIME, 
	update_time DATETIME,
	PRIMARY KEY (id)
);

CREATE INDEX ix_views_id ON views (id);
CREATE INDEX ix_views_name ON views (name);
CREATE INDEX ix_views_virtual_folder ON views (virtual_folder);
CREATE INDEX ix_views_view_type ON views (view_type);

CREATE TABLE issues (
	id INTEGER NOT NULL, 
	keywords VARCHAR,
	status VARCHAR,
	creator VARCHAR,
	view_id INTEGER,
	associate_nodes VARCHAR,
	description VARCHAR, 
	create_time DATETIME, 
	update_time DATETIME,
	PRIMARY KEY (id)
);

CREATE INDEX ix_issues_id ON issues (id);
CREATE INDEX ix_issues_keywords ON issues (keywords);
CREATE INDEX ix_issues_creator ON issues (creator);
CREATE INDEX ix_issues_status ON issues (status);
CREATE INDEX ix_issues_view_id ON issues (view_id);
COMMIT;

