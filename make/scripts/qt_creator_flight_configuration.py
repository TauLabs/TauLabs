'''
Overwrites the Qt Creator user project files to customize for the firmware build environment
'''

# Imports
from os import path
from os import getcwd
from argparse import ArgumentParser

# Create argument parser
parser = ArgumentParser(description='Process board targets.')
parser.add_argument('--targets', metavar='target_board', nargs='+', help='List of board names')

# Parse arguments
args = parser.parse_args()
targets = args.targets

# Check that there are some targets. If not, print help and exit.
if not targets:
	print ""
	print "*******************************"
	print "* No --target arguments found *"
	print "*******************************"
	print ""
	parser.print_help()
	exit(1)

# Set up paths and file names
qt_creator_flight_project_path = "flight/Project/Qt Creator/"
root_directory = getcwd()

file_template = path.join(qt_creator_flight_project_path, "flight.pro.user.template")
file_out      = path.join(qt_creator_flight_project_path, "flight.pro.user")

# Template text to be replaced
dummy_directory = "{ROOT REPOSITORY PATH}"
dummy_configuration_number_text = "{CONFIGURATION NUMBER}"
dummy_target_name_normal = "{TARGET NAME, NORMAL CASE}"
dummy_target_name_lowercase = "{TARGET NAME, LOWERCASE}"
dummy_number_of_targets = "{NUMBER OF TARGETS}"
dummy_build_configuration_targets = "{BUILD CONFIGURATION TARGETS}"

# Generic target configuration block
reference_target_configuration = """   <valuemap type="QVariantMap" key="ProjectExplorer.Target.BuildConfiguration.{CONFIGURATION NUMBER}">
    <value type="QString" key="ProjectExplorer.BuildConfiguration.BuildDirectory">{ROOT REPOSITORY PATH}/build</value>
    <valuemap type="QVariantMap" key="ProjectExplorer.BuildConfiguration.BuildStepList.0">
     <valuemap type="QVariantMap" key="ProjectExplorer.BuildStepList.Step.0">
      <value type="bool" key="ProjectExplorer.BuildStep.Enabled">true</value>
      <value type="QString" key="ProjectExplorer.ProcessStep.Arguments">{TARGET NAME, LOWERCASE}</value>
      <value type="QString" key="ProjectExplorer.ProcessStep.Command">make</value>
      <value type="QString" key="ProjectExplorer.ProcessStep.WorkingDirectory">{ROOT REPOSITORY PATH}</value>
      <value type="QString" key="ProjectExplorer.ProjectConfiguration.DefaultDisplayName">Custom Process Step</value>
      <value type="QString" key="ProjectExplorer.ProjectConfiguration.DisplayName"></value>
      <value type="QString" key="ProjectExplorer.ProjectConfiguration.Id">ProjectExplorer.ProcessStep</value>
     </valuemap>
     <value type="int" key="ProjectExplorer.BuildStepList.StepsCount">1</value>
     <value type="QString" key="ProjectExplorer.ProjectConfiguration.DefaultDisplayName">Build</value>
     <value type="QString" key="ProjectExplorer.ProjectConfiguration.DisplayName"></value>
     <value type="QString" key="ProjectExplorer.ProjectConfiguration.Id">ProjectExplorer.BuildSteps.Build</value>
    </valuemap>
    <valuemap type="QVariantMap" key="ProjectExplorer.BuildConfiguration.BuildStepList.1">
     <valuemap type="QVariantMap" key="ProjectExplorer.BuildStepList.Step.0">
      <value type="bool" key="ProjectExplorer.BuildStep.Enabled">true</value>
      <value type="QString" key="ProjectExplorer.ProjectConfiguration.DefaultDisplayName">Make</value>
      <value type="QString" key="ProjectExplorer.ProjectConfiguration.DisplayName"></value>
      <value type="QString" key="ProjectExplorer.ProjectConfiguration.Id">Qt4ProjectManager.MakeStep</value>
      <valuelist type="QVariantList" key="Qt4ProjectManager.MakeStep.AutomaticallyAddedMakeArguments">
       <value type="QString">-w</value>
       <value type="QString">-r</value>
      </valuelist>
      <value type="bool" key="Qt4ProjectManager.MakeStep.Clean">true</value>
      <value type="QString" key="Qt4ProjectManager.MakeStep.MakeArguments">clean</value>
      <value type="QString" key="Qt4ProjectManager.MakeStep.MakeCommand"></value>
     </valuemap>
     <value type="int" key="ProjectExplorer.BuildStepList.StepsCount">1</value>
     <value type="QString" key="ProjectExplorer.ProjectConfiguration.DefaultDisplayName">Clean</value>
     <value type="QString" key="ProjectExplorer.ProjectConfiguration.DisplayName"></value>
     <value type="QString" key="ProjectExplorer.ProjectConfiguration.Id">ProjectExplorer.BuildSteps.Clean</value>
    </valuemap>
    <value type="int" key="ProjectExplorer.BuildConfiguration.BuildStepListCount">2</value>
    <value type="bool" key="ProjectExplorer.BuildConfiguration.ClearSystemEnvironment">false</value>
    <valuelist type="QVariantList" key="ProjectExplorer.BuildConfiguration.UserEnvironmentChanges"/>
    <value type="QString" key="ProjectExplorer.ProjectConfiguration.DefaultDisplayName">Debug</value>
    <value type="QString" key="ProjectExplorer.ProjectConfiguration.DisplayName">{TARGET NAME, NORMAL CASE}</value>
    <value type="QString" key="ProjectExplorer.ProjectConfiguration.Id">Qt4ProjectManager.Qt4BuildConfiguration</value>
    <value type="int" key="Qt4ProjectManager.Qt4BuildConfiguration.BuildConfiguration">2</value>
    <value type="bool" key="Qt4ProjectManager.Qt4BuildConfiguration.UseShadowBuild">true</value>
   </valuemap>
"""

dummy_build_configuration_count = """   <value type="int" key="ProjectExplorer.Target.BuildConfigurationCount">{NUMBER OF TARGETS}</value>"""


# Creaty empty string
output_target_configuration = ""

# Loop through all board targets
for idx,target_name in enumerate(targets):
    # Make local copy of reference text
    target_configuration = reference_target_configuration

    # Incrementally update local target configuration
    target_configuration = target_configuration.replace(dummy_configuration_number_text, repr(idx))
    target_configuration = target_configuration.replace(dummy_target_name_normal, target_name)
    target_configuration = target_configuration.replace(dummy_target_name_lowercase, target_name.lower())

    # Update output target configuration with local target
    output_target_configuration = output_target_configuration + target_configuration

# Append number of build configurations
output_target_configuration = output_target_configuration + dummy_build_configuration_count.replace(dummy_number_of_targets, repr(len(targets)))

# Open template file and read into memory
f = open(file_template,'r')
filedata = f.read()
f.close()

# Replace template placeholder by target configurations
filedata = filedata.replace(dummy_build_configuration_targets, output_target_configuration)

# Replace dummy directory by actual directory. This must follow the target configuration placeholder replacement
filedata = filedata.replace(dummy_directory, root_directory)

# Write file
f = open(file_out,'w')
f.write(filedata)
f.close()
