/*****PLEASE ENTER YOUR DETAILS BELOW*****/
--T1-rm-schema.sql

--Student ID: 
--Student Name: 

/* Comments for your marker:




*/

/* drop table statements - do not remove*/

DROP TABLE competitor CASCADE CONSTRAINTS PURGE;

DROP TABLE entry CASCADE CONSTRAINTS PURGE;

DROP TABLE team CASCADE CONSTRAINTS PURGE;

/* end of drop table statements*/

-- Task 1 Add Create table statements for the Missing TABLES below.
-- Ensure all column comments, and constraints (other than FK's)are included.
-- FK constraints are to be added at the end of this script

-- COMPETITOR
CREATE TABLE competitor (
    comp_no        NUMBER(5) NOT NULL,
    comp_fname     VARCHAR2(30),
    comp_lname     VARCHAR2(30),
    comp_gender    CHAR(1) NOT NULL CHECK (comp_gender IN ('M','F','U')),
    comp_dob       DATE NOT NULL,
    comp_email     VARCHAR2(50) NOT NULL,
    comp_unistatus CHAR(1) NOT NULL CHECK (comp_unistatus IN ('Y','N')),
    comp_phone     CHAR(10) NOT NULL
);
ALTER TABLE competitor ADD CONSTRAINT comp_pk PRIMARY KEY ( comp_no );

ALTER TABLE competitor ADD CONSTRAINT comp_nk UNIQUE ( comp_email,
                                                       comp_phone );



--ENTRY
CREATE TABLE entry (
    event_id          NUMBER(6) NOT NULL,
    entry_no          NUMBER(5) NOT NULL,
    entry_starttime   TIMESTAMP,
    entry_finishtime  TIMESTAMP,         
    entry_elapsedtime INTERVAL DAY TO SECOND,
    comp_no           NUMBER(5) NOT NULL,
    team_id           NUMBER(3),         
    char_id           NUMBER(3)          
);
ALTER TABLE entry ADD CONSTRAINT entry_pk PRIMARY KEY ( event_id, entry_no );

--TEAM
CREATE TABLE team (
    team_id   NUMBER(3) NOT NULL,
    team_name VARCHAR2(30) NOT NULL,
    carn_date DATE NOT NULL,              
    event_id  NUMBER(6) NOT NULL,         
    entry_no  NUMBER(5) NOT NULL          
);

ALTER TABLE team ADD CONSTRAINT team_pk PRIMARY KEY ( team_id );

ALTER TABLE team ADD CONSTRAINT team_uk UNIQUE ( team_name, carn_date );


-- Add all missing FK Constraints below here

--- ENTRY ---
ALTER TABLE entry
    ADD CONSTRAINT entry_event_fk FOREIGN KEY ( event_id )
        REFERENCES event ( event_id );

ALTER TABLE entry
    ADD CONSTRAINT entry_competitor_fk FOREIGN KEY ( comp_no )
        REFERENCES competitor ( comp_no );

ALTER TABLE entry
    ADD CONSTRAINT entry_team_fk FOREIGN KEY ( team_id )
        REFERENCES team ( team_id );

ALTER TABLE entry
    ADD CONSTRAINT entry_charity_fk FOREIGN KEY ( char_id )
        REFERENCES charity ( char_id );

--- TEAM ---
ALTER TABLE team
    ADD CONSTRAINT team_event_fk FOREIGN KEY ( event_id )
        REFERENCES event ( event_id );

ALTER TABLE team
    ADD CONSTRAINT team_entry_fk FOREIGN KEY ( event_id, entry_no )
        REFERENCES entry ( event_id, entry_no );