release-patch:
	poetry version --next-phase patch
	bumpver update -p

release-minor:
